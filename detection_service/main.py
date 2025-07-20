import pika
import cv2
import base64
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from detector import ViolationDetector
from logic import ViolationLogic
from PIL import Image, ImageDraw
import os
import sys

DISPLAY_DEBUG_STREAM = False

# Paths and config
# MODEL_PATH = "../model/best.pt" # yolo12m-v2
# ROI_CONFIG = "roi_config.json"
# VIOLATION_DIR = "../violations/"
# Path(VIOLATION_DIR).mkdir(exist_ok=True, parents=True)
# RABBITMQ_HOST = "localhost"
# FRAMES_QUEUE = "frames"
# STREAMING_QUEUE = "streaming"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
MODEL_PATH = config.MODEL_PATH
ROI_CONFIG = config.ROI_CONFIG
RABBITMQ_HOST = config.RABBITMQ_HOST
FRAMES_QUEUE = config.FRAMES_QUEUE
STREAMING_QUEUE = config.STREAMING_QUEUE
VIOLATION_DIR = config.VIOLATION_DIR
Path(VIOLATION_DIR).mkdir(exist_ok=True, parents=True)

# Init detector and logic
detector = ViolationDetector(MODEL_PATH, conf_threshold=0.2)
logic = ViolationLogic(ROI_CONFIG)

def decode_frame(frame_b64):
    jpg_data = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(jpg_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def draw_boxes(frame, detections):
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im)
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det['label']
        color = "red" if label == "hand" else "blue"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, y1-10), label, fill=color)
    return im

def save_violation_frame(frame, detections, frame_index):
    img = draw_boxes(frame, detections)
    fname = f"{VIOLATION_DIR}violation_{frame_index}.jpg"
    img.save(fname)
    return fname

def send_to_streaming(channel, data):
    channel.basic_publish(
        exchange='',
        routing_key=STREAMING_QUEUE,
        body=json.dumps(data)
    )

def callback(ch, method, properties, body):
    try:
        msg = json.loads(body)
        frame_index = msg["frame_index"]
        frame_b64 = msg["frame_data"]
        timestamp = msg["timestamp"]
        video_source = msg["video_source"]

        frame = decode_frame(frame_b64)
        detections = detector.detect(frame)
        violation, debug_info = logic.process(detections)

        frame_vis = frame.copy()

        # Draw ingredient/cargo ROIs from config
        for roi in logic.ingredient_rois:
            x1, y1, x2, y2 = roi['x1'], roi['y1'], roi['x2'], roi['y2']
            color = (0, 255, 0)
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame_vis, f"ROI: {roi['name']}", 
                (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

        # Draw dynamic detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            if label == "hand":
                color = (0, 0, 255)
            elif label == "pizza":
                color = (0, 255, 255)
            elif label == "scooper":
                color = (255, 0, 255)
            else:
                color = (255, 0, 0)
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_vis, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if violation:
            cv2.putText(frame_vis, "VIOLATION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        if DISPLAY_DEBUG_STREAM:
            cv2.imshow("Detection Debug", frame_vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting debug window.")
                cv2.destroyAllWindows()
                exit(0)

        result = {
            "frame_index": frame_index,
            "timestamp": timestamp,
            "video_source": video_source,
            "detections": detections,
            "violation": violation,
            "rois": logic.ingredient_rois # .rois
            # "frame_data": frame_b64 
        }

        if violation:
            fname = save_violation_frame(frame, detections, frame_index)
            print(f"[!] Violation detected on frame {frame_index} (saved {fname})")
            result["violation_frame_path"] = fname

        send_to_streaming(ch, result)
        print(f"[>] Processed frame {frame_index} (violation={violation})")
    except Exception as e:
        print("[!] Exception in detection callback:", e)
        import traceback; traceback.print_exc()

def main():
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
        channel = connection.channel()
        channel.queue_declare(queue=FRAMES_QUEUE)
        channel.queue_declare(queue=STREAMING_QUEUE)
        channel.basic_consume(queue=FRAMES_QUEUE, on_message_callback=callback, auto_ack=True)
        print("[*] Waiting for frames. To exit press CTRL+C")
        channel.start_consuming()
    except KeyboardInterrupt:
        print(" [!] Interrupted by user.")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
