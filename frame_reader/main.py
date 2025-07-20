import cv2
import pika
import base64
import json
import datetime
import sys
import os

# -------------------------- CONFIG -------------------------------------
# from config import MODEL_PATH, VIDEO_PATH, RABBITMQ_HOST, TARGET_FPS, RABBITMQ_QUEUE
# VIDEO_PATH = "../data/Sah b3dha ghalt.mp4"  
# RABBITMQ_HOST = "localhost"
# RABBITMQ_QUEUE = "frames"
# TARGET_FPS = 3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
VIDEO_PATH = config.VIDEO_PATH
RABBITMQ_HOST = config.RABBITMQ_HOST
RABBITMQ_QUEUE = config.RABBITMQ_QUEUE
TARGET_FPS = config.TARGET_FPS


# -------------------------- Helper Functions -------------------------------------
# connect to the rabbbit server to upload frames
def connect_rabbitmq():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=RABBITMQ_QUEUE)
    return connection, channel

# encode the frames to base 64
def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def main():
    connection, channel = connect_rabbitmq()
    print("connected to rabbit server ")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"!!!! can not open video file: {VIDEO_PATH}")
        return

    # get the actual fps
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 30  # fallback

    print(f"... processing video: {VIDEO_PATH} with {video_fps} FPS and sending with {TARGET_FPS} FPS")

    frame_index = 0
    sent_index = 0
    frame_interval = int(round(video_fps / TARGET_FPS))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # send reduced number of frames    
        if frame_index % frame_interval == 0:
            timestamp = datetime.datetime.now(datetime.UTC).isoformat() + "Z"
            frame_b64 = encode_frame(frame)
            message = {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "video_source": VIDEO_PATH,
                "frame_data": frame_b64,
            }
            channel.basic_publish(
                exchange='',
                routing_key=RABBITMQ_QUEUE,
                body=json.dumps(message)
            )
            print(f".. sent actual frame {frame_index} as frame number {sent_index} with size = {len(json.dumps(message)) // 1024} KB")
            sent_index += 1

        frame_index += 1

    cap.release()
    connection.close()
    print("# DONE #")

if __name__ == "__main__":
    main()
