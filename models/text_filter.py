import cv2
import numpy as np
import argparse
import time
from paddleocr import PaddleOCR

DEMO_MODE = "QUALITY" // Use "FAST" for better FPS

if DEMO_MODE == "QUALITY":
    PROCESSING_WIDTH = 960
    DETECTION_INTERVAL = 12
    DETECTION_RESOLUTION_WIDTH = 1200
    DETECTION_THRESHOLD = 0.20
    BLUR_STRENGTH = 91
else:
    PROCESSING_WIDTH = 854
    DETECTION_INTERVAL = 15
    DETECTION_RESOLUTION_WIDTH = 720
    DETECTION_THRESHOLD = 0.3
    BLUR_STRENGTH = 81

OUTPUT_FILENAME = "blurred_output_final.mp4"
WEBCAM_WIDTH = 1920
WEBCAM_HEIGHT = 1080

print(f"Initializing models in {DEMO_MODE} mode...")
ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=True, show_log=False, det_db_thresh=DETECTION_THRESHOLD)
print("✅ PaddleOCR model loaded successfully.")

def preprocess_for_detection(frame):
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened_frame = cv2.filter2D(frame, -1, sharpen_kernel)
    return sharpened_frame

def main(input_source):
    cap = cv2.VideoCapture(input_source)
    if isinstance(input_source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {input_source}")
        return

    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    aspect_ratio = original_h / original_w
    processing_h = int(PROCESSING_WIDTH * aspect_ratio)
    print(f"✅ Capturing at {original_w}x{original_h}, processing at {PROCESSING_WIDTH}x{processing_h}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, original_fps, (original_w, original_h))

    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    trackers = None
    current_boxes = []

    while True:
        success, original_frame = cap.read()
        if not success: break
        
        frame = cv2.resize(original_frame, (PROCESSING_WIDTH, processing_h))
        
        if frame_count % DETECTION_INTERVAL == 0:
            print(f"Frame {frame_count}: Running detection...")
            
            detection_w = min(DETECTION_RESOLUTION_WIDTH, PROCESSING_WIDTH)
            detection_h = int(detection_w * aspect_ratio)
            detection_frame = cv2.resize(frame, (detection_w, detection_h))

            preprocessed_detection_frame = preprocess_for_detection(detection_frame)
            
            result = ocr.ocr(preprocessed_detection_frame, cls=False, det=True, rec=False)
            
            trackers = cv2.legacy.MultiTracker_create()
            current_boxes = []
            
            if result and result[0]:
                scale_x = PROCESSING_WIDTH / detection_w
                scale_y = processing_h / detection_h
                for box in result[0]:
                    points = (np.array(box) * [scale_x, scale_y]).astype(np.int32)
                    x, y, w, h = cv2.boundingRect(points)
                    tracker = cv2.legacy.TrackerCSRT_create()
                    trackers.add(tracker, frame, (x, y, w, h))
            
            current_boxes = [tuple(map(int, box)) for box in trackers.getObjects()]

        else:
            if trackers:
                success, new_boxes = trackers.update(frame)
                if success:
                    current_boxes = [tuple(map(int, box)) for box in new_boxes]
        
        if current_boxes:
            blurred_frame = cv2.GaussianBlur(frame, (BLUR_STRENGTH, BL_STRENGTH), 0)
            mask = np.zeros(frame.shape[:2], dtype="uint8")
            for (x, y, w, h) in current_boxes:
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            frame = np.where(mask[:, :, None] == 255, blurred_frame, frame)

        fps_end_time = time.time()
        fps = 1 / (fps_end_time - fps_start_time) if (fps_end_time - fps_start_time) > 0 else 0
        fps_start_time = fps_end_time
        cv2.putText(frame, f"FPS: {fps:.1f} ({DEMO_MODE})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        display_frame = cv2.resize(frame, (original_w, original_h))
        writer.write(display_frame)
        cv2.imshow("Streamer Privacy Shield", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
            
        frame_count += 1

    print("Cleaning up...")
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Privacy Shield.")
    parser.add_argument('video_path', type=str, nargs='?', default='0', 
                        help="Path to a video file. Leave empty to use webcam.")
    args = parser.parse_args()
    input_source = int(args.video_path) if args.video_path.isdigit() else args.video_path
    main(input_source)
