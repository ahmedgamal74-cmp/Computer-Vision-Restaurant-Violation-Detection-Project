import cv2
import json
import sys
import os

# VIDEO_PATH = "../data/Sah b3dha ghalt.mp4"  
# OUTPUT_ROI_PATH = "roi_config.json"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

VIDEO_PATH = config.VIDEO_PATH
OUTPUT_ROI_PATH = config.OUTPUT_ROI_PATH

rois = []
drawing = False
ix, iy = -1, -1
frame = None
temp_frame = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, frame, temp_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_frame = frame.copy()
            cv2.rectangle(temp_frame, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Select ROIs', temp_frame)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        temp_frame = frame.copy()
        cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Select ROIs', temp_frame)
        name = input(f"Enter ROI name for rectangle ({x1},{y1})-({x2},{y2}): ")
        rois.append({
            "name": name,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })
        print(f"ROI '{name}' added. Press 'n' to add another or 's' to save and quit.")

def main():
    global frame, temp_frame
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read the first frame.")
        return

    temp_frame = frame.copy()
    cv2.namedWindow('Select ROIs')
    cv2.setMouseCallback('Select ROIs', draw_rectangle)

    print("Draw each ROI with your mouse (click and drag).")
    print("After each ROI, enter its name in the terminal.")
    print("Press 'n' to draw the next ROI, 's' to save and exit.")

    while True:
        cv2.imshow('Select ROIs', temp_frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            temp_frame = frame.copy()
        elif key == ord('s'):
            break

    cv2.destroyAllWindows()
    # Save ROIs to file
    with open(OUTPUT_ROI_PATH, "w") as f:
        json.dump(rois, f, indent=2)
    print(f"Saved {len(rois)} ROIs to {OUTPUT_ROI_PATH}")

if __name__ == "__main__":
    main()
