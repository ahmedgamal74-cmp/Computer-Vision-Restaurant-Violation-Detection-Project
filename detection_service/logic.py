import json

def load_rois(path="roi_config.json"):
    with open(path, "r") as f:
        return json.load(f)

def bbox_center_in_roi(bbox, roi):
    """Checks if the center of bbox is inside the ROI."""
    x1, y1, x2, y2 = bbox
    rx1, ry1, rx2, ry2 = roi['x1'], roi['y1'], roi['x2'], roi['y2']
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

def bbox_center_in_bbox(bbox, target_bbox):
    """Checks if the center of bbox is inside the target bbox."""
    x1, y1, x2, y2 = bbox
    tx1, ty1, tx2, ty2 = target_bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return tx1 <= cx <= tx2 and ty1 <= cy <= ty2

class ViolationLogic:
    def __init__(self, roi_path="roi_config.json"):
        self.ingredient_rois = load_rois(roi_path)  # All user ROIs are ingredient/cargo zones
        self.rois = load_rois(roi_path)
        
    def process(self, detections):
        violation = False
        hand_in_ingredient = False
        hand_in_pizza = False
        scooper_in_hand = False

        # Find all pizza bounding boxes
        pizza_bboxes = [det['bbox'] for det in detections if det['label'] == "pizza"]

        for det in detections:
            if det['label'] == "hand":
                # Check all static ROIs
                for roi in self.ingredient_rois:
                    if bbox_center_in_roi(det['bbox'], roi):
                        hand_in_ingredient = True
                # Check dynamic pizza zones
                for pizza_bbox in pizza_bboxes:
                    if bbox_center_in_bbox(det['bbox'], pizza_bbox):
                        hand_in_pizza = True
            if det['label'] == "scooper":
                # Check if scooper is overlapping with any hand (distance < 40 px)
                for det2 in detections:
                    if det2['label'] == "hand":
                        cx1, cy1 = (det['bbox'][0]+det['bbox'][2])//2, (det['bbox'][1]+det['bbox'][3])//2
                        cx2, cy2 = (det2['bbox'][0]+det2['bbox'][2])//2, (det2['bbox'][1]+det2['bbox'][3])//2
                        dist = ((cx1-cx2)**2 + (cy1-cy2)**2)**0.5
                        if dist < 50:
                            scooper_in_hand = True

        # Violation: hand in ingredient, then hand in pizza, without scooper in hand
        if hand_in_ingredient and hand_in_pizza and not scooper_in_hand:
            violation = True

        debug_info = {
            "hand_in_ingredient": hand_in_ingredient,
            "hand_in_pizza": hand_in_pizza,
            "scooper_in_hand": scooper_in_hand,
            "ingredient_rois": [r["name"] for r in self.ingredient_rois],
            "pizza_bboxes": pizza_bboxes,
        }
        return violation, debug_info

# For direct testing:
if __name__ == "__main__":
    detections = [
        {"bbox": [120, 160, 170, 210], "label": "hand", "conf": 0.95},
        {"bbox": [120, 160, 170, 210], "label": "scooper", "conf": 0.92},
        {"bbox": [400, 250, 450, 300], "label": "hand", "conf": 0.90},
        {"bbox": [390, 240, 460, 310], "label": "pizza", "conf": 0.93},
    ]
    logic = ViolationLogic("roi_config.json")
    violation, info = logic.process(detections)
    print("Violation:", violation)
    print("Debug info:", info)
