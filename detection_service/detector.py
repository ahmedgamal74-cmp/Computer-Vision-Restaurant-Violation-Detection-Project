from ultralytics import YOLO
import numpy as np
import cv2

class ViolationDetector:
    def __init__(self, model_path, conf_threshold=0.3):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """
        Args:
            frame (numpy.ndarray): BGR image from cv2
        Returns:
            List of detections: 
            [{"bbox": [x1, y1, x2, y2], "label": "hand", "conf": 0.89}, ...]
        """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(img, conf=self.conf_threshold)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label_idx = int(box.cls[0])
            label = results[0].names[label_idx]
            conf = float(box.conf[0])
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "conf": conf
            })
        return detections

# # For direct testing:
# if __name__ == "__main__":
#     model_path = "../model/yolo12m-v2.pt"
#     detector = ViolationDetector(model_path, conf_threshold=0.1)
#     video_path = "../data/wrong.mp4"
#     cap = cv2.VideoCapture(video_path)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         results = detector.detect(frame)
#         print(results)
#     cap.release()


