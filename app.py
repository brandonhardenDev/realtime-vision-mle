from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import uvicorn
import io
from PIL import Image

app = FastAPI(title="Vision-MLE API")

# Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

@app.get("/")
async def health_check():
    return {"status": "healthy", "model": "YOLOv8n"}

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Run inference
    results = model.predict(image)
    
    # Process results
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "box": box.xyxy.tolist()
            })
            
    return {"detections": detections}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
