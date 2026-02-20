# 🛣️ Smart Pothole Detection System — Optimized Edition

## Hardware Stack
| Component | Model | Interface |
|-----------|-------|-----------|
| Single-board Computer | Raspberry Pi 4B | — |
| LiDAR Sensor | Benewake TF-02 Pro | UART `/dev/ttyAMA4` @ 115200 |
| GPS Module | u-blox NEO-6M | UART `/dev/ttyAMA0` @ 9600 |

> **No camera, no ultrasonic, no GSM, no Bluetooth** — streamlined to the three hardware components specified.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                Raspberry Pi 4B                      │
│                                                     │
│  TF-02 Pro ──► SensorHub ──► SlidingWindowBuffer   │
│  NEO-6M    ──►    │              │                  │
│                   │         DepthImageBuilder       │
│                   │              │                  │
│                   │         YOLOv8Detector  ◄──────── yolov8_pothole.pt
│                   │              │                  │
│                   │         OnlineTrainer  (bg)     │
│                   │              │                  │
│                   └──► BackendClient (HTTP + WS)    │
└─────────────────────────────────────────────────────┘
                          │
                    Cloud/LAN Backend
                    (FastAPI + SQLite)
                          │
                    Web Dashboard
               (Leaflet Map + Live WebSocket)
```

## Key Features
- **Real-time YOLOv8 detection** on 2D depth-image slices generated from rolling LiDAR buffer
- **Online continual learning**: every confirmed detection is auto-labelled and used to fine-tune the model in a background thread (no reboot needed)
- **Zero extra hardware**: only the three components listed above
- **Async FastAPI backend** with live WebSocket push to dashboard
- **Live admin panel** to view/repair potholes, monitor model versions, and download training logs

---

## Directory Structure
```
pothole_detection/
├── raspi/
│   ├── config.py            – System configuration dataclass
│   ├── sensors.py           – LiDAR + GPS drivers (no legacy ultrasonic)
│   ├── yolo_detector.py     – YOLOv8 inference on depth frames
│   ├── online_trainer.py    – Background continual-learning engine
│   ├── depth_image.py       – Rolling buffer → 2D depth image conversion
│   ├── backend_client.py    – HTTP + retry upload to FastAPI
│   ├── main.py              – Main async orchestrator (entry point)
│   ├── requirements.txt
│   └── setup.sh
├── backend/
│   ├── main.py              – FastAPI backend (WebSocket, REST, DB)
│   ├── requirements.txt
│   └── schema.sql
├── dashboard/
│   ├── index.html           – Live map dashboard
│   ├── style.css
│   └── app.js
└── README.md
```

---

## Quick Start (Raspberry Pi)
```bash
# 1. Install dependencies
cd raspi && bash setup.sh

# 2. Run the detection system
python main.py

# 3. Run the backend (separate terminal or service)
cd ../backend && uvicorn main:app --host 0.0.0.0 --port 8000
```

## Pin / UART Connections
| Signal | Pi GPIO | Pi UART |
|--------|---------|---------|
| LiDAR TX → Pi RX | GPIO 9 (Pin 21) | `/dev/ttyAMA4` |
| LiDAR RX ← Pi TX | GPIO 8 (Pin 24) | `/dev/ttyAMA4` |
| GPS TX → Pi RX | GPIO 15 (Pin 22) | `/dev/ttyAMA0` |
| GPS GND | GND | — |
| LiDAR VCC | 5V | — |
| GPS VCC | 3.3V | — |
