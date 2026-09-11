from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import io
import base64

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "microplastic-detection-yolo8m.pt"

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")

print("Loading YOLO model...")
model = YOLO(str(MODEL_PATH))
print("YOLO model loaded.")

@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")

@app.post("/api/detect")
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No image file received"}), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = uploaded.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        try:
            conf = float(request.form.get("conf", "0.25"))
        except ValueError:
            conf = 0.25
        threshold = max(0.01, min(0.95, conf))

        results = model.predict(
            source=image,
            conf=threshold,
            imgsz=640,
            verbose=False
        )

        result = results[0]
        detections = []

        if result.boxes is not None:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = [round(float(v), 2) for v in box.xyxy[0].tolist()]

                detections.append({
                    "class_id": cls_id,
                    "class_name": str(names.get(cls_id, cls_id)),
                    "confidence": round(conf, 4),
                    "box": xyxy
                })

        # Render YOLO bounding boxes onto the uploaded image.
        plotted_bgr = result.plot()
        plotted_rgb = plotted_bgr[:, :, ::-1]
        rendered = Image.fromarray(plotted_rgb)

        buffer = io.BytesIO()
        rendered.save(buffer, format="JPEG", quality=90)
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        avg_confidence = (
            sum(d["confidence"] for d in detections) / len(detections)
            if detections else 0
        )

        return jsonify({
            "count": len(detections),
            "detections": detections,
            "average_confidence": round(avg_confidence, 4),
            "threshold": threshold,
            "image": "data:image/jpeg;base64," + image_b64
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False)
