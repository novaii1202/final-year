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


def _normalize_name(s: str) -> str:
    """Lowercase and strip spaces/specials for stable name comparison."""
    return (
        s.strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace("(", "")
        .replace(")", "")
    )


# ─────────────────────────────────────────────────────
# Threaded Video Capture (important for Raspberry Pi)
# ─────────────────────────────────────────────────────
class VideoCaptureAsync:
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


# ─────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────
# Load valid tube classes
# ─────────────────────────────────────────────────────
def load_valid_classes(data_dir: Path):

    classes_file = data_dir / "classes.txt"

    if not classes_file.exists():
        return []

    with open(classes_file, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f.readlines() if line.strip()]


# ─────────────────────────────────────────────────────
# Main System
# ─────────────────────────────────────────────────────
def run_system(source_input,
               imgsz=320,
               min_conf=0.9,
               frame_skip=2,
               min_box_area=4000,
               nms_threshold=0.7,
               dedupe=False,
               debug=False,
               headless=False):

    base_dir = Path(__file__).resolve().parent.parent

    # Use the latest trained weights from brand_experiment3
    onnx_model_path = base_dir / "runs/detect/brand_experiment3/weights/best.onnx"
    ncnn_model_path = base_dir / "runs/detect/brand_experiment3/weights/best_ncnn_model"
    tflite_model_path = base_dir / "runs/detect/brand_experiment3/weights/best_saved_model/best_float32.tflite"
    pt_model_path = base_dir / "runs/detect/brand_experiment3/weights/best.pt"

    # Clamp to a high-confidence operating point
    if min_conf < 0.9:
        print(f"🔧 Bumping min_conf from {min_conf:.2f} to 0.90 for high-confidence tube-only detections.")
        min_conf = 0.9

    # ───── Load model ─────
    if onnx_model_path.exists():
        print("🚀 Loading ONNX model")
        model = YOLO(str(onnx_model_path), task='detect')

    elif ncnn_model_path.exists():
        print("🚀 Loading NCNN model")
        model = YOLO(str(ncnn_model_path), task='detect')

    elif tflite_model_path.exists():
        print("🚀 Loading TFLite model")
        model = YOLO(str(tflite_model_path), task='detect')

    elif pt_model_path.exists():
        print("✅ Loading PyTorch model")
        model = YOLO(str(pt_model_path))

    else:
        raise RuntimeError(
            "No trained tube-detection model found in runs/detect/brand_experiment3/weights.\n"
            "Please place your latest best.pt / ONNX / NCNN / TFLite weights there and retry."
        )

    # ───── Load valid classes ─────
    valid_classes = load_valid_classes(base_dir / "data" / "dataset(tubes)")
    normalized_valid = {_normalize_name(c) for c in valid_classes}

    # Map model label variants to canonical class names from classes.txt
    alias_map = {
        # Model label '(Valbet)' → classes.txt 'valbet'
        "valbet": "valbet",
        # Model label 'S K Kant' → classes.txt 'silverkant'
        "skkant": "silverkant",
    }

    db_path = base_dir / "data/inspections.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Text log file for every detected tube
    log_file = base_dir / "data" / "detections.log"

    conn = setup_db(db_path)

    # ───── Camera setup ─────
    if source_input.isnumeric():
        source_input = int(source_input)
        cap = VideoCaptureAsync(source_input)
        print("📹 Using threaded capture")

    else:
        cap = cv2.VideoCapture(source_input)

    logged_objects = set()
    total_tubes_logged = 0  # running count: +1 each time a tube is logged to DB

    frame_count = 0

    fps_start = time.time()
    fps_counter = 0
    display_fps = 0.0

    print("🎥 Camera Active. Press 'Q' to quit.")

    if not headless:
        cv2.namedWindow("ML Brand Detector", cv2.WINDOW_NORMAL)

    # ─────────────────────────────
    # Main loop
    # ─────────────────────────────
    while cap.isOpened():

        success, frame = cap.read()

        if not success or frame is None:
            break

        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        # ───── YOLO Tracking (IMPORTANT FIX) ─────
        with torch.no_grad():

            results = model.track(
                frame,
                conf=min_conf,
                imgsz=imgsz,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

        detections = []

        if results[0].boxes is not None and len(results[0].boxes) > 0:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().numpy()
            else:
                track_ids = np.full(len(boxes), -1, dtype=int)

            for box, cls, conf, track_id in zip(boxes, clss, confs, track_ids):

                x1, y1, x2, y2 = box

                raw_name = model.names[int(cls)]
                norm_model_name = _normalize_name(raw_name)
                # Resolve to canonical label if we know an alias
                canonical = alias_map.get(norm_model_name, raw_name.lower())
                if _normalize_name(canonical) not in normalized_valid:
                    continue

                area = (x2 - x1) * (y2 - y1)

                if area < min_box_area:
                    continue

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                detections.append((canonical, conf, (x1, y1, x2, y2), track_id))

                if not headless:

                    # Yellow box around detected tube
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                    label = f"ID:{track_id} {canonical}"

                    # Yellow label text
                    cv2.putText(frame,
                                label,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2)

        # ───── Logging ─────
        if detections:

            names = [d[0] for d in detections]
            track_ids = [d[3] for d in detections]

            unique_ids = {tid for tid in track_ids if tid != -1}

            for brand_name, conf, _, track_id in detections:

                if conf < min_conf:
                    continue

                if dedupe:
                    key = (track_id, brand_name)

                    if key in logged_objects:
                        continue

                    logged_objects.add(key)

                uuid_code = log_inspection(conn, track_id, brand_name, conf)
                total_tubes_logged += 1

                # Append to plain-text log so every detection is recorded
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                try:
                    with open(log_file, "a", encoding="utf-8") as lf:
                        lf.write(f"{timestamp},track={track_id},tube={brand_name},conf={conf:.2f},uuid={uuid_code}\n")
                except Exception:
                    # Avoid breaking the main loop if logging fails
                    pass

                print(f"Logged to DB and file: {brand_name} | Confidence: {conf:.2f}")
            print(f"Detected Tube: {', '.join(names)} | Total tubes: {total_tubes_logged}")

        # ───── FPS counter ─────
        fps_counter += 1

        elapsed = time.time() - fps_start

        if elapsed >= 1.0:
            display_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        cv2.putText(frame,
                    f"FPS: {display_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2)

        if not headless:

            cv2.imshow("ML Brand Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:

            if frame_count % 20 == 0:
                print(f"⚡ FPS: {display_fps:.1f}")

    cap.release()

    if not headless:
        cv2.destroyAllWindows()

    conn.close()

    print(f"\n✅ Session complete. Final FPS: {display_fps:.1f}")


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--min-conf", type=float, default=0.85)
    parser.add_argument("--min-area", type=int, default=4000)
    parser.add_argument("--nms-thresh", type=float, default=0.7)
    parser.add_argument("--skip", type=int, default=2)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--headless", action="store_true")

    args = parser.parse_args()

    run_system(
        args.source,
        imgsz=args.imgsz,
        min_conf=args.min_conf,
        frame_skip=args.skip,
        min_box_area=args.min_area,
        nms_threshold=args.nms_thresh,
        dedupe=args.dedupe,
        debug=args.debug,
        headless=args.headless,
    )