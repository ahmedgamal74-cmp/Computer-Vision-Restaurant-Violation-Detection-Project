# 🍕 Computer Vision Restaurant Violation Detection Project

A modular, microservices-based system for **real-time detection of food handling violations in restaurant kitchens**, powered by computer vision (YOLO), RabbitMQ, and a live browser UI.

**Repo:** [ahmedgamal74-cmp/Computer-Vision-Restaurant-Violation-Detection-Project](https://github.com/ahmedgamal74-cmp/Computer-Vision-Restaurant-Violation-Detection-Project)

---

## Features

- Microservices architecture: Frame Reader, Detection, Streaming/UI
- Real-time detection with YOLO (custom or pre-trained models)
- Live violation logic using user-defined ROIs
- Scalable messaging with RabbitMQ
- Modern browser UI with video, overlays, violation counter
- Configuration in `config.py` and `.gitignore`
- (Optional) Docker Compose for one-command deployment

---

## Architecture Overview
```
┌──────────────┐     ┌──────────────┐    ┌──────────────┐     ┌───────────────┐
│ Frame Reader │ ───▶ │ Detection   │───▶ │ Streaming  │ ───▶ │ Web Browser │ 
│ (Video → MQ) │     │ (YOLO, MQ) │ │    │ (API/WS/UI)  │     │ (UI + API)    │
└──────────────┘     └──────────────┘    └──────────────┘     └───────────────┘
```
- **Frame Reader:** Sends video frames to RabbitMQ.
- **Detection:** Detects objects, applies violation logic, publishes results.
- **Streaming/UI:** Streams annotated video & serves live stats to browser.

---

## Directory Structure
```
├── detection_service/
│ ├── main.py
│ ├── detector.py
│ ├── logic.py
│ ├── config.py
│ └── ...
├── frame_reader/
│ ├── main.py
│ └── ...
├── streaming_service/
│ ├── main.py
│ └── ...
├── model/ # (ignored) YOLO weights
├── data/ # (ignored) Video/data files
├── config.py
├── roi_config.json
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Microservices

### 1. Frame Reader
- Reads video frames at set FPS, sends to RabbitMQ.

### 2. Detection Service
- Runs YOLO detection and applies violation logic using ROIs.
- Publishes results to streaming queue.

### 3. Streaming Service/UI
- Streams annotated video frames via WebSocket.
- REST API for live violation stats and ROI info.
- Browser UI overlays detections & ROIs, shows violation count.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ahmedgamal74-cmp/Computer-Vision-Restaurant-Violation-Detection-Project.git
cd Computer-Vision-Restaurant-Violation-Detection-Project
```

### 2. Install Requirements

```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate # On Mac/Linux
pip install -r requirements.txt
```

### 3. Start RabbitMQ (Recommended: Docker)

```bash
docker run -d --hostname pizza-mq --name pizza-rabbit -p 5672:5672 -p 15672:15672 rabbitmq:3-management
```

### 4. Prepare Model and Video
- Place your YOLO .pt weights in model/
- Place a video in data/ (e.g., data/wrong.mp4)

### 5. Configure the Project
- Edit config.py as needed.
- Use the ROI selector tool or edit roi_config.json manually.

### 6. Run the Services
- Each service runs in its own terminal or process:

```bash
# Frame Reader
cd frame_reader
python main.py

# Detection Service
cd detection_service
python main.py

# Streaming/UI Service
cd streaming_service
uvicorn main:app --host 0.0.0.0 --port 5000
```

- Open http://localhost:5000 in your browser.

## Usage
- Browser UI displays video with detection overlays, ROIs, and live violation count.
### API Endpoints:
  - `/stats` — live violation count (JSON)
  - `/roi_config` — ROI info (JSON)
  - `/ws` — WebSocket for video frames

## License

MIT License
