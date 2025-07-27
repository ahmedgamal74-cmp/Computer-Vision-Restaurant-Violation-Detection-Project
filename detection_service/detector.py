from ultralytics import YOLO
import numpy as np
import cv2

class ViolationDetector:
    def __init__(self, model_path, conf_threshold=0.3):
        self.model = YOLO(model_path).to('cuda')
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """
        Args: frame image from cv2
        Returns: detections:  box [x1, y1, x2, y2], label, conf
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


