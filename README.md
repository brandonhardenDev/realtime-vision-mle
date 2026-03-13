# Real-time Object Tracking & MLOps Pipeline

This repository contains a production-ready Computer Vision pipeline designed for real-time object detection and tracking. Inspired by mission-critical security applications, it leverages **YOLOv8** for inference and **FastAPI** for serving.

## ðŸš€ Key Features
- **Real-time Inference:** Optimized YOLOv8 pipeline for high-throughput video stream analysis.
- **RESTful API:** Robust FastAPI endpoints for model interaction and health monitoring.
- **MLOps Integration:** Dockerized environment for seamless deployment across edge and cloud.
- **Asynchronous Processing:** Utilizes Python's `asyncio` for non-blocking I/O operations.

## ðŸ›  Tech Stack
- **Deep Learning:** Ultralytics YOLOv8, PyTorch
- **API Framework:** FastAPI, Uvicorn
- **Deployment:** Docker
- **Tracking:** BoT-SORT / ByteTrack

## ðŸ“¦ Project Structure
- `app/`: FastAPI application code.
- `models/`: Model weights and configuration.
- `Dockerfile`: Containerization setup.
- `requirements.txt`: Python dependencies.
