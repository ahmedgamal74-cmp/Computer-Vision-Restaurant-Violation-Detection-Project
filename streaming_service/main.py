import cv2
import pika
import json
import threading
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
import asyncio
import os
import sys

# ------------------------------- Configrations -------------------------
# VIDEO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/Sah w b3dha ghalt (2).mp4"))
# RABBITMQ_HOST = "localhost"
# STREAMING_QUEUE = "streaming"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
VIDEO_PATH = config.VIDEO_PATH
RABBITMQ_HOST = config.RABBITMQ_HOST
STREAMING_QUEUE = config.STREAMING_QUEUE
VIDEO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), VIDEO_PATH))

app = FastAPI()

# CORS for local testing and API calls from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve the frontend (HTML/JS)  from FastAPI 
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/frontend", StaticFiles(directory=frontend_dir, html=True), name="frontend")


latest_detection = {}
violation_count = 0
lock = threading.Lock()

# upload streaming frames to rabbit
def rabbitmq_consumer():
    global latest_detection, violation_count
    def callback(ch, method, properties, body):
        global violation_count
        msg = json.loads(body)
        with lock:
            latest_detection.update(msg)
            if msg.get("violation", False):
                violation_count += 1
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=STREAMING_QUEUE)
    channel.basic_consume(queue=STREAMING_QUEUE, on_message_callback=callback, auto_ack=True)
    print(" waiting for detection results ......")
    channel.start_consuming()

# ------------------------------------- endpoints ------------------------------
# endpoint to get number of ddetections 
@app.get("/stats")
async def get_stats():
    global violation_count
    with lock:
        return JSONResponse({
            "total_violations": violation_count
        })
# endpoint to send ROIs json
@app.get("/roi_config")
async def get_roi_config():
    import json
    with open("../detection_service/roi_config.json", "r") as f:  
        rois = json.load(f)
    return rois
# endpoint for web
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(VIDEO_PATH)
    prev_frame_idx = -1
    try:
        while True:
            with lock:
                frame_idx = latest_detection.get("frame_index")
                detections = latest_detection.get("detections", [])
                violation = latest_detection.get("violation", False)
            if frame_idx is not None and frame_idx != prev_frame_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    await websocket.send_bytes(b"")  # Empty frame
                    await asyncio.sleep(0.03)
                    continue
                # Draw overlays
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    label = det['label']
                    color = (0, 0, 255) if label == "hand" else (0, 255, 255) if label == "pizza" else (255, 0, 255) if label == "scooper" else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if violation:
                    cv2.putText(frame, "VIOLATION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                _, jpeg = cv2.imencode('.jpg', frame)
                await websocket.send_bytes(jpeg.tobytes())
                prev_frame_idx = frame_idx
            await asyncio.sleep(0.03)
    except WebSocketDisconnect:
        cap.release()
        print("WebSocket client disconnected")

# start the rabbit consumer
threading.Thread(target=rabbitmq_consumer, daemon=True).start()
