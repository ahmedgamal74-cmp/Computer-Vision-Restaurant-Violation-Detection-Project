import os 

VIDEO_PATH = "../data/Sah w b3dha ghalt (2).mp4" 
# VIDEO_PATH_ABS = os.path.abspath(os.path.join(os.path.dirname(__file__), VIDEO_PATH))

MODEL_PATH = "../model/best3.pt"      # "../model/yolo12m-v2.pt"
ROI_CONFIG = "roi_config.json"
VIOLATION_DIR = "../violations/"

RABBITMQ_HOST = "localhost"
FRAMES_QUEUE = "frames"
STREAMING_QUEUE = "streaming"
RABBITMQ_QUEUE = "frames"

OUTPUT_ROI_PATH = "roi_config.json"
TARGET_FPS = 10