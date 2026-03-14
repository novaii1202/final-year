import cv2
import argparse
import uuid
import datetime
import sqlite3
import time
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from ultralytics import YOLO

# ───────────────────────────────────────────────────────────────────
# OpenVINO-optimized inference for Raspberry Pi 4
# Intel OpenVINO now supports ARM CPUs via pip wheels.
# This script is tuned for maximum FPS on Pi with OpenVINO backend.
# ───────────────────────────────────────────────────────────────────

class VideoCaptureAsync:
    """Threaded video capture — keeps frame grabbing off the inference thread."""
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.running = True
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            grabbed, frame = self.cap.read()
            if grabbed:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        return self.grabbed, self.frame

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


def setup_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inspections (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            track_id INTEGER,
            brand_name TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    return conn

def log_inspection(conn, track_id, class_name, conf):
    check_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO inspections (id, timestamp, track_id, brand_name, confidence)
        VALUES (?, ?, ?, ?, ?)
    ''', (check_id, timestamp, track_id, class_name, round(conf, 2)))
    conn.commit()
    return check_id

def load_valid_classes(data_dir: Path):
    classes_file = data_dir / "classes.txt"
    if not classes_file.exists():
        return []
    with open(classes_file, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f.readlines() if line.strip()]


def run_system(source_input, imgsz=320, min_conf=0.85, frame_skip=2, min_box_area=2000, max_box_area=None, nms_threshold=0.7, dedupe=False, debug=False, headless=False):
    base_dir = Path(__file__).resolve().parent.parent

    # ── Model paths (OpenVINO first priority here) ──
    openvino_model_path = base_dir / "runs/detect/brand_experiment32/weights/best_openvino_model"
    onnx_model_path = base_dir / "runs/detect/brand_experiment32/weights/best.onnx"
    ncnn_model_path = base_dir / "runs/detect/brand_experiment32/weights/best_ncnn_model"
    pt_model_path = base_dir / "runs/detect/brand_experiment32/weights/best.pt"

    # ── Load Model (OpenVINO preferred) ──
    if openvino_model_path.exists():
        print(f"🚀 Loading OpenVINO model (320x320): {openvino_model_path}")
        model = YOLO(str(openvino_model_path), task='detect')
    elif onnx_model_path.exists():
        print(f"🔄 Falling back to ONNX model: {onnx_model_path}")
        model = YOLO(str(onnx_model_path), task='detect')
    elif ncnn_model_path.exists():
        print(f"🔄 Falling back to NCNN model: {ncnn_model_path}")
        model = YOLO(str(ncnn_model_path), task='detect')
    elif pt_model_path.exists():
        print(f"🔄 Falling back to PyTorch model: {pt_model_path}")
        model = YOLO(str(pt_model_path))
    else:
        print("⚠️ No custom model found! Using standard YOLOv11n.")
        model = YOLO("yolo11n.pt")

    valid_classes = load_valid_classes(base_dir / "data" / "dataset(tubes)")
    if not valid_classes:
        print("⚠️ Warning: No valid classes loaded from data/dataset(tubes)/classes.txt")

    db_path = base_dir / "data/inspections.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = setup_db(db_path)

    # ── Video capture ──
    if source_input.isnumeric():
        source_input = int(source_input)
        cap = VideoCaptureAsync(source_input)
        print("📹 Using threaded video capture for max FPS")
    else:
        cap = cv2.VideoCapture(source_input)

    logged_objects = set()
    frame_count = 0

    # FPS tracking
    fps_start = time.time()
    fps_counter = 0
    display_fps = 0.0

    # Calculate Max Area based on frame if not provided
    # tubes shouldn't usually cover more than 70% of the frame
    if max_box_area is None:
        max_box_area = (imgsz * imgsz) * 0.7  # Default to 70% of inference size

    print(f"🎥 Camera Active. Filter: {min_box_area} < Area < {max_box_area}")
    print("Press 'Q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None:
            break

        frame_count += 1
        # Skip frames for FPS improvement
        if frame_count % frame_skip != 0:
            continue

        h, w = frame.shape[:2]
        # Resize frame to reduce processing load
        small_frame = cv2.resize(frame, (imgsz, imgsz), interpolation=cv2.INTER_AREA)

        # Inference with tracking
        with torch.no_grad():
            results = model.track(small_frame, conf=min_conf, imgsz=imgsz, persist=True, verbose=False)

        detections = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # Scale factor for boxes back to original frame size
            scale_w, scale_h = w / imgsz, h / imgsz
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().numpy()
            else:
                track_ids = np.full(len(boxes), -1, dtype=int)

            # Build list for NMS (x, y, w, h) in inference coordinates
            bboxes_xywh = []
            for x1, y1, x2, y2 in boxes:
                bboxes_xywh.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

            idxs = cv2.dnn.NMSBoxes(bboxes_xywh, confs.tolist(), min_conf, nms_threshold)
            idxs = idxs.flatten().tolist() if len(idxs) > 0 else []

            for idx in idxs:
                # Raw coordinates from inference
                rx1, ry1, rx2, ry2 = boxes[idx]
                
                # Check area in inference space
                area = (rx2 - rx1) * (ry2 - ry1)
                if area < min_box_area or area > max_box_area:
                    continue

                # Project back to original frame size
                x1, y1 = int(rx1 * scale_w), int(ry1 * scale_h)
                x2, y2 = int(rx2 * scale_w), int(ry2 * scale_h)
                
                cls = int(clss[idx])
                conf = float(confs[idx])
                track_id = int(track_ids[idx])
                brand_name = model.names[cls].lower()

                # Filter: only allow known tube classes
                if brand_name not in valid_classes:
                    continue

                detections.append((brand_name, conf, (x1, y1, x2, y2), track_id))

                if not headless:
                    color = (0, 255, 0) # Green
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{brand_name} {conf:.2f}"
                    if track_id != -1:
                        label_text = f"#{track_id} {label_text}"
                    cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Persistence & Gap Logic
        current_time = time.time()
        if not hasattr(run_system, 'logged_ids'):
            run_system.logged_ids = set()
        if not hasattr(run_system, 'last_action_time'):
            run_system.last_action_time = 0

        # GLOBAL GAP: Check if enough time has passed since the LAST log of ANY object
        # This prevents rapid-fire logging of multiple objects appearing at once
        global_gap = 5.0 # 5 seconds gap after every detection
        time_since_last_log = current_time - run_system.last_action_time

        if detections and time_since_last_log > global_gap:
            for brand_name, conf, _, track_id in detections:
                # 1. Tracked Object Logic (ONLY store if track_id exists)
                if track_id != -1:
                    if track_id in run_system.logged_ids:
                        continue # Already logged this specific object
                    
                    run_system.logged_ids.add(track_id)
                    uuid_code = log_inspection(conn, track_id, brand_name, conf)
                    run_system.last_action_time = current_time
                    print(f"📝 [LOGGED] {brand_name} (ID:{track_id}). UUID: {uuid_code}. Next log available in {global_gap}s")
                    break # Log only ONE thing per gap interval
                
                # If track_id == -1, we skip logging as per requirements

        # FPS counter
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            display_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        # Overlay global info
        cv2.rectangle(frame, (0, 0), (220, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"THR: {min_conf:.2f}", (10, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if not headless:
            cv2.imshow("ML Brand Detector [OpenVINO]", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # Real-time tuning keys
            elif key == ord('+'):
                min_conf = min(0.95, min_conf + 0.05)
            elif key == ord('-'):
                min_conf = max(0.1, min_conf - 0.05)
        else:
            if frame_count % 20 == 0:
                print(f"⚡ OpenVINO FPS: {display_fps:.1f}")

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    conn.close()
    print(f"\n✅ Session complete. Final FPS: {display_fps:.1f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Camera ID (0) or video path")
    parser.add_argument("--imgsz", type=int, default=320, help="Size for inference")
    parser.add_argument("--min-conf", type=float, default=0.85, help="Min confidence")
    parser.add_argument("--min-area", type=int, default=1500, help="Min area")
    parser.add_argument("--max-area", type=int, default=None, help="Max area (filters faces/big objects)")
    parser.add_argument("--nms-thresh", type=float, default=0.7, help="NMS threshold")
    parser.add_argument("--skip", type=int, default=2, help="Skip every Nth frame")
    parser.add_argument("--dedupe", action="store_true", help="Log unique track_id once")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    args = parser.parse_args()
    run_system(
        args.source,
        imgsz=args.imgsz,
        min_conf=args.min_conf,
        frame_skip=args.skip,
        min_box_area=args.min_area,
        max_box_area=args.max_area,
        nms_threshold=args.nms_thresh,
        dedupe=args.dedupe,
        debug=args.debug,
        headless=args.headless,
    )

