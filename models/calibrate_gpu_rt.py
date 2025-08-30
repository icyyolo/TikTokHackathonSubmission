import cv2
import os
os.environ['CUDA_PATH'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
import numpy as np
import sqlite3
from datetime import datetime
import time
import argparse
import sys
# --- TensorRT & CUDA Dependencies ---
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from numpy.linalg import norm

# --- Configuration ---
MODELS_DIR = "models"
DET_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8m-face-lindevs.onnx")
REC_MODEL_PATH = os.path.join(MODELS_DIR, "arcface_r100_v1.onnx")

# --- IMPROVED: Set more realistic quality thresholds ---
MIN_FACE_SIZE = 48  # Minimum face size in pixels (height or width)
QUALITY_THRESHOLD = 40.0 # Laplacian variance, a measure of blurriness
CONF_THRESHOLD = 0.6 # Confidence threshold for face detection
NMS_THRESHOLD = 0.4 # Non-Maximum Suppression threshold

DATABASE_PATH = "streamer_embeddings.db"

# A helper class to store face information
class Face:
    def __init__(self, bbox, landmark, det_score, embedding):
        self.bbox = bbox
        self.landmark = landmark
        self.det_score = det_score
        self.embedding = embedding

# --- TensorRT Engine & Inference ---
class TensorRTFaceAnalysis:
    """
    Handles face detection (YOLOv8-Face) and recognition (ArcFace) using TensorRT.
    """
    def __init__(self, det_onnx_path, rec_onnx_path):
        self.logger = trt.Logger(trt.Logger.WARNING) # Changed to WARNING for cleaner logs
        self.det_engine_path = det_onnx_path.replace('.onnx', '.engine')
        self.rec_engine_path = rec_onnx_path.replace('.onnx', '.engine')

        self.det_engine = self._load_or_build_engine(det_onnx_path, self.det_engine_path)
        self.rec_engine = self._load_or_build_engine(rec_onnx_path, self.rec_engine_path)

        if not self.det_engine or not self.rec_engine:
            raise RuntimeError("Failed to load or build one or more TensorRT engines.")

        self.det_context = self.det_engine.create_execution_context()
        self.rec_context = self.rec_engine.create_execution_context()
        
        # Get input shape for the detection model dynamically
        det_input_name = self.det_engine.get_tensor_name(0)
        self.det_input_shape = self.det_engine.get_tensor_shape(det_input_name)
        self.det_input_height = self.det_input_shape[2]
        self.det_input_width = self.det_input_shape[3]

        # Standard reference landmarks for ArcFace alignment
        self.ref_landmarks = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]
        ], dtype=np.float32)

    def _load_or_build_engine(self, onnx_path, engine_path):
        if os.path.exists(engine_path):
            print(f"✅ Loading existing TensorRT engine: {engine_path}")
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            print(f"⚠️ Engine not found. Building from ONNX: {onnx_path}")
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"ONNX model not found at {onnx_path}. Please check the path.")

            builder = trt.Builder(self.logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            config = builder.create_builder_config()
            parser = trt.OnnxParser(network, self.logger)

            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # 2GB

            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(f"Parser Error: {parser.get_error(error)}")
                    raise ValueError(f"Failed to parse the ONNX file: {onnx_path}")

            profile = builder.create_optimization_profile()
            input_name = network.get_input(0).name

            if onnx_path == DET_MODEL_PATH:
                shape = (1, 3, 640, 640)
                profile.set_shape(input_name, min=shape, opt=shape, max=shape)
            elif onnx_path == REC_MODEL_PATH:
                shape = (1, 112, 112, 3)
                profile.set_shape(input_name, min=shape, opt=shape, max=shape)

            config.add_optimization_profile(profile)
            print(f"Building TensorRT engine for {os.path.basename(onnx_path)}. This may take a few minutes...")
            serialized_engine = builder.build_serialized_network(network, config)

            if serialized_engine is None:
                raise RuntimeError(f"Failed to build TensorRT engine for {onnx_path}")

            print("Engine built successfully. Saving to disk.")
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)

            runtime = trt.Runtime(self.logger)
            return runtime.deserialize_cuda_engine(serialized_engine)

    def _preprocess_yolo(self, img):
        input_w, input_h = self.det_input_width, self.det_input_height
        h, w, _ = img.shape
        
        # Calculate scaling factor
        scale = min(input_w / w, input_h / h)
        resized_w, resized_h = int(w * scale), int(h * scale)
        
        # Resize image with calculated scale
        resized_img = cv2.resize(img, (resized_w, resized_h))
        
        # Create a blank canvas (padded image)
        padded_img = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        
        # Place resized image at the top-left corner
        padded_img[:resized_h, :resized_w] = resized_img
        
        # Normalize and transpose for the model (HWC to CHW)
        img_data = np.transpose(padded_img, (2, 0, 1)).astype(np.float32) / 255.0
        img_data = np.expand_dims(img_data, axis=0)
        
        return img_data, scale, w, h

    def _postprocess_yolo(self, output, scale, original_w, original_h):
        """
        Correctly post-processes YOLOv8-Face output and applies Non-Maximum Suppression.
        Output format is [x_center, y_center, width, height, confidence]
        """
        # Transpose output from (1, 5, 8400) or (5, 8400) to (8400, 5)
        if len(output.shape) == 3:
            output = output[0]
        if output.shape[0] == 5 and output.shape[1] > 5:
             output = output.transpose(1, 0)
        
        # Filter out detections with low confidence
        scores = output[:, 4]
        high_conf_indices = scores > CONF_THRESHOLD
        
        if not np.any(high_conf_indices):
            return np.array([]), np.array([]), np.array([])
            
        output = output[high_conf_indices]
        scores = scores[high_conf_indices]
        
        # Extract boxes
        boxes = output[:, :4]
        
        # --- THIS IS THE FIX ---
        # The padding is only on the right and bottom. 
        # We only need to divide by the scale factor to get coordinates on the original image.
        # The previous code was incorrectly subtracting a padding value, shifting the box up.
        boxes /= scale
        
        # Convert from (center_x, center_y, width, height) to (x1, y1, x2, y2)
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        # --- NEW: Apply Non-Maximum Suppression (NMS) ---
        # This prevents detecting multiple overlapping boxes for the same face.
        # Note: cv2.dnn.NMSBoxes expects boxes as (x, y, w, h)
        xywh_boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in boxes]
        nms_indices = cv2.dnn.NMSBoxes(xywh_boxes, scores.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)
        
        if len(nms_indices) == 0:
            return np.array([]), np.array([]), np.array([])
            
        final_boxes = np.array([x1, y1, x2, y2]).T[nms_indices]
        final_scores = scores[nms_indices]

        # --- NEW: Clip bounding boxes to be within image dimensions ---
        final_boxes[:, 0] = np.clip(final_boxes[:, 0], 0, original_w - 1)
        final_boxes[:, 1] = np.clip(final_boxes[:, 1], 0, original_h - 1)
        final_boxes[:, 2] = np.clip(final_boxes[:, 2], 0, original_w - 1)
        final_boxes[:, 3] = np.clip(final_boxes[:, 3], 0, original_h - 1)

        # Estimate landmarks based on the final bounding boxes
        num_detections = len(final_boxes)
        landmarks = np.zeros((num_detections, 5, 2))
        for i, (bx1, by1, bx2, by2) in enumerate(final_boxes):
            face_w, face_h = bx2 - bx1, by2 - by1
            landmarks[i] = np.array([
                [bx1 + 0.3 * face_w, by1 + 0.4 * face_h],  # Left eye
                [bx1 + 0.7 * face_w, by1 + 0.4 * face_h],  # Right eye
                [bx1 + 0.5 * face_w, by1 + 0.6 * face_h],  # Nose tip
                [bx1 + 0.35 * face_w, by1 + 0.8 * face_h], # Left mouth corner
                [bx1 + 0.65 * face_w, by1 + 0.8 * face_h]  # Right mouth corner
            ])
            
        return final_boxes, landmarks, final_scores

    def _align_face(self, img, landmark):
        tform = cv2.estimateAffinePartial2D(landmark, self.ref_landmarks)[0]
        return cv2.warpAffine(img, tform, (112, 112), borderValue=0.0)

    def _preprocess_rec(self, face_img):
        img_data = face_img.astype(np.float32)
        img_data = np.expand_dims(img_data, axis=0)
        return (img_data - 127.5) * 0.0078125

    def get(self, img):
        # --- Detection (YOLOv8-Face) ---
        img_preprocessed, scale, original_w, original_h = self._preprocess_yolo(img)

        det_input_name = self.det_engine.get_tensor_name(0)
        det_output_name = self.det_engine.get_tensor_name(1)
        det_output_shape = self.det_engine.get_tensor_shape(det_output_name)
        
        d_input = cuda.mem_alloc(img_preprocessed.nbytes)
        h_output = np.empty(det_output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(h_output.nbytes)
        stream = cuda.Stream()

        self.det_context.set_tensor_address(det_input_name, int(d_input))
        self.det_context.set_tensor_address(det_output_name, int(d_output))

        cuda.memcpy_htod_async(d_input, np.ascontiguousarray(img_preprocessed), stream)
        self.det_context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        bboxes, landmarks, scores = self._postprocess_yolo(h_output, scale, original_w, original_h)
        
        d_input.free()
        d_output.free()
        
        # --- Recognition (ArcFace) ---
        faces = []
        if len(bboxes) == 0:
            return faces

        for bbox, landmark, score in zip(bboxes, landmarks, scores):
            aligned_face = self._align_face(img, landmark)
            rec_input = self._preprocess_rec(aligned_face)

            rec_input_name = self.rec_engine.get_tensor_name(0)
            rec_output_name = self.rec_engine.get_tensor_name(1)
            rec_output_shape = self.rec_engine.get_tensor_shape(rec_output_name)

            d_rec_input = cuda.mem_alloc(rec_input.nbytes)
            h_rec_output = np.empty(rec_output_shape, dtype=np.float32)
            d_rec_output = cuda.mem_alloc(h_rec_output.nbytes)
            rec_stream = cuda.Stream()
            
            self.rec_context.set_tensor_address(rec_input_name, int(d_rec_input))
            self.rec_context.set_tensor_address(rec_output_name, int(d_rec_output))

            cuda.memcpy_htod_async(d_rec_input, np.ascontiguousarray(rec_input), rec_stream)
            self.rec_context.execute_async_v3(stream_handle=rec_stream.handle)
            cuda.memcpy_dtoh_async(h_rec_output, d_rec_output, rec_stream)
            rec_stream.synchronize()
            
            embedding = h_rec_output.flatten()
            embedding = embedding / np.linalg.norm(embedding)
            faces.append(Face(bbox=bbox, landmark=landmark, det_score=score, embedding=embedding))
            
            d_rec_input.free()
            d_rec_output.free()
            
        return faces

class VideoCalibrationProcessor:
    """Processes calibration videos using TensorRT-accelerated models."""
    def __init__(self):
        self.face_app = None
        self.conn = None
        self.setup_models()
        self.setup_database()

    def setup_models(self):
        try:
            print("🔄 Loading TensorRT Face Analysis models...")
            start_time = time.time()
            self.face_app = TensorRTFaceAnalysis(DET_MODEL_PATH, REC_MODEL_PATH)
            print(f"✅ Models loaded successfully in {time.time() - start_time:.2f}s")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise

    def setup_database(self):
        try:
            self.conn = sqlite3.connect(DATABASE_PATH)
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS streamers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, streamer_id TEXT UNIQUE NOT NULL, name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    mean_embedding BLOB NOT NULL, embedding_dimension INTEGER, face_count INTEGER,
                    video_hash TEXT, metadata TEXT
                )''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, streamer_id TEXT NOT NULL, embedding BLOB NOT NULL,
                    face_image BLOB, quality_score REAL, extraction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    bbox_x1 INTEGER, bbox_y1 INTEGER, bbox_x2 INTEGER, bbox_y2 INTEGER,
                    FOREIGN KEY (streamer_id) REFERENCES streamers (streamer_id)
                )''')
            self.conn.commit()
            print(f"✅ Database initialized: {DATABASE_PATH}")
        except Exception as e:
            print(f"❌ Error setting up database: {e}")
            if self.conn:
                self.conn.close()
            raise

    def assess_face_quality(self, face_image):
        issues = []
        if face_image.size == 0: return 0.0, False, ["Empty image"]
        if face_image.shape[0] < MIN_FACE_SIZE or face_image.shape[1] < MIN_FACE_SIZE:
            issues.append("Too small")
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < QUALITY_THRESHOLD: issues.append(f"Blurry (score: {lap_var:.1f})")
        is_acceptable = not issues
        return lap_var, is_acceptable, issues

    def process_calibration_video(self, video_path, streamer_id, streamer_name=None, append_mode=True, save_debug_images=False):
        print(f"🎥 Processing video: {video_path} for Streamer ID: {streamer_id}")
        if not os.path.exists(video_path): return {"success": False, "error": "Video not found"}

        existing_faces = self.get_streamer_face_data(streamer_id) if append_mode else []

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps / 2)) if fps > 0 else 1
        new_faces, frame_count = [], 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if frame_count % frame_interval == 0:
                detected_faces = self.face_app.get(frame)

                if not detected_faces:
                    continue 

                # Focus on the largest face for calibration
                face = max(detected_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

                bbox = face.bbox.astype(np.int32)
                x1, y1, x2, y2 = np.maximum(0, bbox)
                face_crop = frame[y1:y2, x1:x2]
                quality_score, is_acceptable, issues = self.assess_face_quality(face_crop)

                if is_acceptable:
                    new_faces.append({
                        'face_crop': face_crop, 'embedding': face.embedding,
                        'quality_score': quality_score, 'bbox': bbox.tolist()
                    })
                    print(f"  ✓ Face {len(new_faces)} extracted (quality: {quality_score:.1f})")
                    if save_debug_images:
                        os.makedirs(f"debug_{streamer_id}", exist_ok=True)
                        cv2.imwrite(f"debug_{streamer_id}/face_{len(new_faces)}.png", face_crop)
                else:
                    print(f"  ✗ Face rejected. Reason(s): {', '.join(issues)}")
                    if save_debug_images:
                        debug_dir = f"debug_rejected_{streamer_id}"
                        os.makedirs(debug_dir, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        cv2.imwrite(os.path.join(debug_dir, f"rejected_{timestamp}.png"), face_crop)
            frame_count += 1
        cap.release()

        if not new_faces: return {"success": False, "error": "No acceptable faces found"}
        
        all_faces = existing_faces + new_faces
        mean_embedding = np.mean([f['embedding'] for f in all_faces], axis=0)
        mean_embedding /= np.linalg.norm(mean_embedding)
        self.save_to_database(streamer_id, streamer_name, mean_embedding, all_faces)
        return {"success": True, "total_faces": len(all_faces)}

    def get_streamer_face_data(self, streamer_id):
        cursor = self.conn.cursor()
        cursor.execute('SELECT embedding, quality_score, bbox_x1, bbox_y1, bbox_x2, bbox_y2 FROM face_embeddings WHERE streamer_id = ?', (streamer_id,))
        return [{'embedding': np.frombuffer(row[0], dtype=np.float32), 'bbox': [row[2],row[3],row[4],row[5]]} for row in cursor.fetchall()]

    def save_to_database(self, streamer_id, streamer_name, mean_embedding, face_data):
        cursor = self.conn.cursor()
        cursor.execute('''INSERT OR REPLACE INTO streamers 
            (streamer_id, name, mean_embedding, face_count, updated_at) VALUES (?, ?, ?, ?, ?)''', 
            (streamer_id, streamer_name or streamer_id, mean_embedding.tobytes(), 
            len(face_data), datetime.now().isoformat()))
        cursor.execute('DELETE FROM face_embeddings WHERE streamer_id = ?', (streamer_id,))
        for face in face_data:
            image_blob = cv2.imencode('.jpg', face['face_crop'])[1].tobytes() if 'face_crop' in face else None
            cursor.execute('''INSERT INTO face_embeddings 
                (streamer_id, embedding, face_image, quality_score, bbox_x1, bbox_y1, bbox_x2, bbox_y2) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
                (streamer_id, face['embedding'].tobytes(), image_blob, face.get('quality_score'), 
                face['bbox'][0], face['bbox'][1], face['bbox'][2], face['bbox'][3]))
        self.conn.commit()
        print(f"✅ Saved {len(face_data)} faces for streamer {streamer_id} to database.")

    def close(self):
        if self.conn:
            self.conn.close()
            print("Database connection closed.")


def main(video_path, streamer_id, streamer_name=None, append_mode=True, save_debug_images=False):
    processor = None
    try:
        processor = VideoCalibrationProcessor()
        result = processor.process_calibration_video(
            video_path, streamer_id, streamer_name, append_mode, save_debug_images
        )
        if result["success"]:
            print(f"🎉 Calibration successful! Total of {result['total_faces']} faces processed.")
        else:
            print(f"❌ Calibration failed: {result['error']}")
    except Exception as e:
        print(f"An unhandled error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if processor:
            processor.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video to calibrate a streamer's face embedding.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--streamer_id", type=str, required=True, help="Unique identifier for the streamer.")
    parser.add_argument("--streamer_name", type=str, default=None, help="Display name for the streamer.")
    parser.add_argument("--no_append", action="store_false", dest="append_mode", help="Overwrite existing embeddings instead of appending.")
    parser.add_argument("--save_debug_images", action="store_true", help="Save extracted and rejected face crops to disk.")

    args = parser.parse_args()

    main(
        video_path=args.video_path,
        streamer_id=args.streamer_id,
        streamer_name=args.streamer_name,
        append_mode=args.append_mode,
        save_debug_images=args.save_debug_images
    )
