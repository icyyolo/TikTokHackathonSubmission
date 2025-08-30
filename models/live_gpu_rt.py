import cv2
import numpy as np
import time
import threading
from collections import deque
import sqlite3
import os
os.environ['CUDA_PATH'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4" # DO NOT REMOVE
import queue
from numpy.linalg import norm
from numpy import dot

# TensorRT & CUDA Dependencies
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Configuration
CUDA_DEVICE_ID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA_DEVICE_ID)

MODELS_DIR = "models"
DET_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8m-face-lindevs.onnx")
REC_MODEL_PATH = os.path.join(MODELS_DIR, "arcface_r100_v1.onnx")

BUFFER_DELAY_SECONDS = 5.0
RECOGNITION_THRESHOLD = 0.29
CONFIDENCE_THRESHOLD = 0.6
BLUR_KERNEL_SIZE = (25, 25)
TARGET_FPS = 60
INPUT_RESOLUTION = (1080, 1920)  # Width, Height
DATABASE_PATH = "streamer_embeddings.db"
STREAMER_ID_TO_LOAD = "mx"

class TensorRTYOLODetector:
    """TensorRT-optimized YOLOv8 face detector"""
    def __init__(self, model_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine_path = model_path.replace('.onnx', '.engine')
        self.engine = self._load_or_build_engine(model_path, self.engine_path)
        self.context = self.engine.create_execution_context()
        
        # Get model bindings
        self.input_binding = None
        self.output_binding = None
        self._setup_bindings()
        
        print(f"TensorRT YOLO detector loaded")
    
    def _load_or_build_engine(self, onnx_path, engine_path):
        """Load existing engine or build from ONNX"""
        if os.path.exists(engine_path):
            print(f"Loading existing TensorRT engine: {engine_path}")
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            print(f"Building TensorRT engine from ONNX: {onnx_path}")
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
            
            builder = trt.Builder(self.logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            config = builder.create_builder_config()
            parser = trt.OnnxParser(network, self.logger)
            
            # TensorRT 10.x uses different memory pool API
            if hasattr(trt.MemoryPoolType, 'WORKSPACE'):
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
            else:
                config.max_workspace_size = 2 << 30  # Fallback for older versions
            
            # Enable FP16 optimization for better performance on modern GPUs
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(f"ONNX Parser Error: {parser.get_error(error)}")
                    raise ValueError("Failed to parse ONNX model")
            
            # TensorRT 10.x uses build_serialized_network
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)
            
            runtime = trt.Runtime(self.logger)
            return runtime.deserialize_cuda_engine(serialized_engine)
    
    def _setup_bindings(self):
        """Setup input/output bindings using modern TensorRT API"""
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_binding = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_binding = name
    
    def _preprocess_image(self, image):
        """Preprocess image for YOLOv8"""
        input_size = 640
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(input_size / w, input_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Convert to float and normalize
        input_data = padded.astype(np.float32) / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)
        
        # Ensure contiguous array for CUDA
        input_data = np.ascontiguousarray(input_data)
        
        return input_data, scale, (new_w, new_h)
    
    def _postprocess_detections(self, outputs, scale, resized_shape):
        """Post-process YOLOv8 detections"""
        # Debug: Print output shape
        print(f"Raw output shape: {outputs.shape}")
        
        predictions = outputs[0]  # Remove batch dimension if needed
        
        # Handle different output formats
        if len(predictions.shape) == 3:
            predictions = predictions[0]
        
        print(f"Predictions shape after processing: {predictions.shape}")
        
        # YOLOv8 output format is typically [batch, 84, 8400] where:
        # - 84 = 4 (bbox) + 1 (conf) + 79 (classes, but we only care about face class)
        # - 8400 = number of anchor points
        
        # If predictions need transposing (common case)
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        
        print(f"Final predictions shape: {predictions.shape}")
        
        # Extract components - YOLOv8-face typically has format [x,y,w,h,conf,landmarks...]
        boxes = predictions[:, :4]  # x_center, y_center, width, height
        scores = predictions[:, 4]  # confidence scores
        
        print(f"Found {len(scores)} detections before filtering")
        print(f"Score range: {scores.min():.3f} to {scores.max():.3f}")
        
        # Filter by confidence
        valid_detections = scores > CONFIDENCE_THRESHOLD
        boxes = boxes[valid_detections]
        scores = scores[valid_detections]
        
        print(f"Found {len(scores)} detections after confidence filtering (threshold: {CONFIDENCE_THRESHOLD})")
        
        if len(scores) == 0:
            return []
        
        # Convert from center format to corner format and scale back
        faces = []
        for i in range(len(scores)):
            x_center, y_center, width, height = boxes[i]
            
            # Scale back to original image
            x_center /= scale
            y_center /= scale
            width /= scale
            height /= scale
            
            # Convert to corner format
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # Get original image dimensions
            original_h = int(resized_shape[1] / scale)
            original_w = int(resized_shape[0] / scale)
            
            # Clamp to image bounds
            x1 = max(0, min(original_w, x1))
            y1 = max(0, min(original_h, y1))
            x2 = max(0, min(original_w, x2))
            y2 = max(0, min(original_h, y2))
            
            # Ensure valid bbox
            if x2 > x1 and y2 > y1:
                faces.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(scores[i])
                })
                print(f"Face {i}: bbox=({x1},{y1},{x2},{y2}), conf={scores[i]:.3f}")
        
        print(f"Returning {len(faces)} valid faces")
        return faces
    
    def detect_faces(self, image):
        """Detect faces in image using TensorRT"""
        input_data, scale, resized_shape = self._preprocess_image(image)
        
        try:
            # Set input shape if dynamic
            input_shape = self.engine.get_tensor_shape(self.input_binding)
            if input_shape[0] == -1:
                self.context.set_input_shape(self.input_binding, input_data.shape)
            
            output_shape = self.context.get_tensor_shape(self.output_binding)
            
            # Allocate GPU memory - fix type casting
            d_input = cuda.mem_alloc(int(input_data.nbytes))
            d_output = cuda.mem_alloc(int(np.prod(output_shape) * 4))  # Convert to int
            
            # Set tensor addresses
            self.context.set_tensor_address(self.input_binding, int(d_input))
            self.context.set_tensor_address(self.output_binding, int(d_output))
            
            # Copy input to GPU and run inference
            stream = cuda.Stream()
            cuda.memcpy_htod_async(d_input, input_data, stream)
            self.context.execute_async_v3(stream_handle=stream.handle)
            
            # Copy output back to CPU
            h_output = np.empty(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()
            
            # Post-process detections
            faces = self._postprocess_detections(h_output, scale, resized_shape)
            return faces
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []

class TensorRTArcFaceRecognizer:
    """TensorRT-optimized ArcFace recognition"""
    def __init__(self, model_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine_path = model_path.replace('.onnx', '.engine')
        self.engine = self._load_or_build_engine(model_path, self.engine_path)
        self.context = self.engine.create_execution_context()
        
        # Get model bindings
        self.input_binding = None
        self.output_binding = None
        self._setup_bindings()
        
        print(f"TensorRT ArcFace recognizer loaded")
    
    def _load_or_build_engine(self, onnx_path, engine_path):
        """Load existing engine or build from ONNX"""
        if os.path.exists(engine_path):
            print(f"Loading existing TensorRT engine: {engine_path}")
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            print(f"Building TensorRT engine from ONNX: {onnx_path}")
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
            
            builder = trt.Builder(self.logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            config = builder.create_builder_config()
            parser = trt.OnnxParser(network, self.logger)
            
            # TensorRT 10.x uses different memory pool API
            if hasattr(trt.MemoryPoolType, 'WORKSPACE'):
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
            else:
                config.max_workspace_size = 2 << 30  # Fallback for older versions
            
            # Enable FP16 optimization for better performance on modern GPUs
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(f"ONNX Parser Error: {parser.get_error(error)}")
                    raise ValueError("Failed to parse ONNX model")
            
            # TensorRT 10.x uses build_serialized_network
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)
            
            runtime = trt.Runtime(self.logger)
            return runtime.deserialize_cuda_engine(serialized_engine)
    
    def _setup_bindings(self):
        """Setup input/output bindings using modern TensorRT API"""
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_binding = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_binding = name
    
    def _preprocess_face(self, face_image):
        """Preprocess face image for ArcFace"""
        target_size = (112, 112)
        
        # Resize face
        resized = cv2.resize(face_image, target_size)
        
        # Normalize to [0, 1] (not [-1, 1])
        normalized = resized.astype(np.float32) / 255.0
        
        # Keep in HWC format and add batch dimension (no transpose!)
        input_data = np.expand_dims(normalized, axis=0)
        
        return input_data
    
    def extract_embedding(self, face_image):
        """Extract face embedding using TensorRT"""
        try:
            input_data = self._preprocess_face(face_image)
            
            # Get output shape
            output_shape = self.context.get_tensor_shape(self.output_binding)
            
            # Allocate GPU memory - fix type casting
            d_input = cuda.mem_alloc(int(input_data.nbytes))
            d_output = cuda.mem_alloc(int(np.prod(output_shape) * 4))  # Convert to int
            
            # Set tensor addresses
            self.context.set_tensor_address(self.input_binding, int(d_input))
            self.context.set_tensor_address(self.output_binding, int(d_output))
            
            # Create a CUDA stream
            stream = cuda.Stream()

            # Copy input to GPU and run inference asynchronously
            cuda.memcpy_htod_async(d_input, input_data, stream)
            self.context.execute_async_v3(stream_handle=stream.handle)

            # Copy output back to CPU asynchronously
            h_output = np.empty(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh_async(h_output, d_output, stream)

            # Wait for the stream to finish all tasks
            stream.synchronize()

            # Normalize embedding
            embedding = h_output.flatten()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Recognition error: {e}")
            return None

class FrameInputQueue:
    """A thread-safe queue to hold incoming frames from the camera."""
    def __init__(self, max_size=150):  # Buffer about 5 seconds of frames
        self.queue = queue.Queue(maxsize=max_size)

    def add_frame(self, frame, timestamp):
        """Adds a frame to the queue without blocking the camera thread."""
        frame_data = {'frame': frame.copy(), 'timestamp': timestamp}
        try:
            self.queue.put_nowait(frame_data)
        except queue.Full:
            # This is "load shedding": if workers are too slow, drop the oldest frame.
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(frame_data)
            except queue.Empty:
                pass  # Should not happen, but safe to handle

    def get_frame(self, timeout=1):
        """Called by worker threads. Blocks until a frame is available."""
        return self.queue.get(timeout=timeout)

    def get_queue_size(self):
        return self.queue.qsize()

class ProcessedFrameBuffer:
    """Buffer for storing individual processed frames, enabling a continuous pipeline."""
    def __init__(self, max_duration=30.0):
        self.frames = deque()
        self.max_frames = int(max_duration * TARGET_FPS)
        self.lock = threading.Lock()

    def add_frame(self, frame_data):
        with self.lock:
            self.frames.append(frame_data)
            while len(self.frames) > self.max_frames:
                self.frames.popleft()
    
    def get_frame_for_timestamp(self, timestamp):
        """Finds the chronologically correct frame for a given timestamp."""
        with self.lock:
            best_frame = None
            min_diff = float('inf')
            for frame_data in self.frames:
                if frame_data['timestamp'] <= timestamp:
                    diff = timestamp - frame_data['timestamp']
                    if diff < min_diff:
                        min_diff = diff
                        best_frame = frame_data
            return best_frame

class FaceRecognitionProcessor:
    """Face recognition processor using TensorRT YOLOv8 + ArcFace"""
    def __init__(self):
        self.streamer_embeddings = {}
        self.context_lock = threading.Lock()  # Add thread safety
        print("Loading TensorRT face recognition models...")
        
        try:
            # Initialize CUDA context in main thread
            cuda.init()
            self.cuda_context = cuda.Device(0).make_context()
            
            self.face_detector = TensorRTYOLODetector(DET_MODEL_PATH)
            self.face_recognizer = TensorRTArcFaceRecognizer(REC_MODEL_PATH)
            print("TensorRT face recognition models loaded successfully")
            
            # Warmup
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.face_detector.detect_faces(dummy_frame)
            print("Model warmup completed")
            
        except Exception as e:
            print(f"Error loading TensorRT models: {e}")
            raise
    
    def load_streamer_embedding(self, streamer_id):
        if not os.path.exists(DATABASE_PATH):
            print(f"Error: Database '{DATABASE_PATH}' not found. Please run the calibration script first.")
            return False
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT mean_embedding FROM streamers WHERE streamer_id = ?", (streamer_id,))
            result = cursor.fetchone()
            conn.close()

            if result and result[0] is not None:
                embedding = np.frombuffer(result[0], dtype=np.float32)
                self.streamer_embeddings[streamer_id] = embedding
                print(f"Loaded mean embedding for streamer: {streamer_id}")
                return True
            else:
                print(f"No embedding found for streamer '{streamer_id}' in the database.")
                return False
        except Exception as e:
            print(f"Error loading streamer embedding from database: {e}")
            return False
    
    def blur_face(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        if x2 <= x1 or y2 <= y1: 
            return frame
        
        face_region = frame[y1:y2, x1:x2].copy()
        if face_region.size == 0: 
            return frame
        
        blurred_face = cv2.GaussianBlur(face_region, BLUR_KERNEL_SIZE, 0)
        
        mask = np.zeros(face_region.shape[:2], dtype=np.uint8)
        center = (face_region.shape[1] // 2, face_region.shape[0] // 2)
        axes = (face_region.shape[1] // 2, face_region.shape[0] // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        mask_3d = np.stack([mask] * 3, axis=-1)
        blended_face = np.where(mask_3d > 0, blurred_face, face_region)
        
        result_frame = frame.copy()
        result_frame[y1:y2, x1:x2] = blended_face
        return result_frame
    
    def process_frame(self, frame, streamer_id):
        if streamer_id not in self.streamer_embeddings: 
            print(f"No embedding found for streamer: {streamer_id}")
            return frame.copy()
        
        # Use context lock to ensure thread safety
        with self.context_lock:
            try:
                # Push context to current thread
                self.cuda_context.push()
                
                faces = self.face_detector.detect_faces(frame)
                print(f"Detected {len(faces)} faces in frame")
                
                result_frame = frame.copy()
                streamer_embedding = self.streamer_embeddings[streamer_id]
                
                for i, face in enumerate(faces):
                    x1, y1, x2, y2 = face['bbox']
                    confidence = face['confidence']
                    
                    print(f"Processing face {i}: bbox=({x1},{y1},{x2},{y2}), conf={confidence:.3f}")
                    
                    # Extract face crop for recognition
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        print(f"Face {i}: Empty crop, skipping")
                        continue
                    
                    print(f"Face {i}: crop size={face_crop.shape}")
                    
                    # Get embedding
                    embedding = self.face_recognizer.extract_embedding(face_crop)
                    if embedding is None:
                        print(f"Face {i}: Failed to extract embedding")
                        continue
                    
                    # Calculate similarity
                    similarity = dot(embedding, streamer_embedding) / (norm(embedding) * norm(streamer_embedding))
                    print(f"Face {i}: similarity={similarity:.3f}, threshold={RECOGNITION_THRESHOLD}")
                    
                    if similarity > RECOGNITION_THRESHOLD:
                        color = (0, 255, 0)
                        label = f"STREAMER: {similarity:.2f}"
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(result_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        print(f"Face {i}: Identified as STREAMER")
                    else:
                        color = (0, 0, 255)
                        label = f"OTHER: {similarity:.2f}"
                        result_frame = self.blur_face(result_frame, [x1, y1, x2, y2])
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(result_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        print(f"Face {i}: Identified as OTHER, applied blur")
                
                return result_frame
            except Exception as e:
                print(f"Error processing frame: {e}")
                import traceback
                traceback.print_exc()
                return frame.copy()
            finally:
                # Pop context from current thread
                try:
                    self.cuda_context.pop()
                except:
                    pass

class VideoProcessingWorker(threading.Thread):
    """A worker thread that continuously processes frames from an input queue."""
    def __init__(self, thread_id, input_queue, processed_buffer, face_processor, system_running_event, streamer_id):
        super().__init__()
        self.thread_id = thread_id
        self.input_queue = input_queue
        self.processed_buffer = processed_buffer
        self.face_processor = face_processor
        self.system_running_event = system_running_event
        self.streamer_id = streamer_id
        self.daemon = True

    def run(self):
        print(f"Worker thread {self.thread_id} started.")
        while self.system_running_event.is_set():
            try:
                frame_data = self.input_queue.get_frame(timeout=1)
                processed_frame = self.face_processor.process_frame(frame_data['frame'], self.streamer_id)
                self.processed_buffer.add_frame({'frame': processed_frame, 'timestamp': frame_data['timestamp']})
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker thread {self.thread_id}: Error: {e}")
        print(f"Worker thread {self.thread_id} stopped.")

class LiveVideoStreamingSystem:
    def __init__(self):
        self.frame_buffer = FrameInputQueue()
        self.processed_buffer = ProcessedFrameBuffer()
        self.face_processor = FaceRecognitionProcessor()
        
        self.workers = []
        # Use only 1 worker for TensorRT to avoid CUDA context conflicts
        self.num_workers = 1
        
        self.system_start_time = None
        self.streamer_id = None
        self.running_event = threading.Event()
        
        self.stats = {'frames_received': 0, 'frames_output': 0}
    
    def start_streaming(self, streamer_id):
        print(f"Starting TensorRT live streaming system for streamer: {streamer_id}")
        if not self.face_processor.load_streamer_embedding(streamer_id):
            return False
        
        self.streamer_id = streamer_id
        self.system_start_time = time.time()
        self.running_event.set()
        
        print(f"Starting {self.num_workers} processing worker threads...")
        for i in range(self.num_workers):
            worker = VideoProcessingWorker(
                thread_id=f"worker_{i+1}",
                input_queue=self.frame_buffer,
                processed_buffer=self.processed_buffer,
                face_processor=self.face_processor,
                system_running_event=self.running_event,
                streamer_id=self.streamer_id
            )
            worker.start()
            self.workers.append(worker)

        print("TensorRT live streaming system started")
        return True
    
    def stop_streaming(self):
        print("Stopping TensorRT live streaming system...")
        self.running_event.clear()
        print("... waiting for worker threads to finish.")
        for worker in self.workers:
            worker.join(timeout=2.0)
        print("TensorRT live streaming system stopped")
    
    def add_frame(self, frame):
        if not self.running_event.is_set(): 
            return
        timestamp = time.time() - self.system_start_time
        self.frame_buffer.add_frame(frame, timestamp)
        self.stats['frames_received'] += 1
    
    def get_processed_frame_for_display(self):
        """Calculates the correct target time and retrieves the corresponding processed frame."""
        if not self.running_event.is_set() or self.system_start_time is None:
            return None, None

        target_output_time = (time.time() - self.system_start_time) - BUFFER_DELAY_SECONDS
        if target_output_time < 0:
            return None, None

        frame_data = self.processed_buffer.get_frame_for_timestamp(target_output_time)
        if frame_data:
            self.stats['frames_output'] += 1
            return frame_data['frame'], frame_data['timestamp']
        return None, None

def test_system():
    system = LiveVideoStreamingSystem()
    if not system.start_streaming(STREAMER_ID_TO_LOAD): 
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_RESOLUTION[0])
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    print("Webcam opened. Press 'q' in the window to quit.")
    
    last_stats_time = time.time()
    
    try:
        while system.running_event.is_set():
            loop_start_time = time.time()
            
            ret, input_frame = cap.read()
            if not ret: 
                break
            
            system.add_frame(input_frame)
            
            output_frame, timestamp = system.get_processed_frame_for_display()
            display_frame = output_frame if output_frame is not None else input_frame
            
            if timestamp is not None:
                cv2.putText(display_frame, f"T: {timestamp:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Resize for display if necessary
            display_h, display_w = 720, 1280
            if display_frame.shape[0] > display_h or display_frame.shape[1] > display_w:
                display_frame = cv2.resize(display_frame, (display_w, display_h))

            cv2.imshow('TensorRT Live Filtered Stream', display_frame)

            if time.time() - last_stats_time > 5.0:
                current_time = time.time() - system.system_start_time
                stats = system.stats
                print(f"Stats at {current_time:.1f}s | Recv: {stats['frames_received']} | Out: {stats['frames_output']} | Threads: {len(system.workers)} | Buf: {system.frame_buffer.get_queue_size()}")
                last_stats_time = time.time()
            
            loop_duration = time.time() - loop_start_time
            sleep_time = (1.0 / TARGET_FPS) - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                system.stop_streaming()
                break
    finally:
        if system.running_event.is_set(): 
            system.stop_streaming()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_system()