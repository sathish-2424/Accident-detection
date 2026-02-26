from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
from datetime import datetime
import cv2
import csv
import os
import pandas as pd


LOG_FILE = "accident_log.csv"
CONFIDENCE_THRESHOLD = 0.5
ACCIDENT_CONF_MIN = 70.0     
ACCIDENT_CLASS_ID = 0


app = Flask(__name__)
model = YOLO("best.pt")
camera = cv2.VideoCapture(0)


state = {
    "status": "Normal",
    "confidence": 0.0,
    "alert_sent": False,
}


def log_accident(conf: float) -> None:
    """Append an accident record to the CSV log, writing the header if needed."""
    file_exists = os.path.isfile(LOG_FILE)
    now = datetime.now()

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date", "Time", "Confidence"])
        writer.writerow([now.date(), now.strftime("%H:%M:%S"), round(conf, 2)])


def detect_accident(results) -> tuple[bool, float]:
    """
    Parse YOLO results and return (accident_detected, highest_confidence).
    Only considers detections where class == ACCIDENT_CLASS_ID and
    confidence exceeds ACCIDENT_CONF_MIN.
    """
    best_conf = 0.0
    detected = False

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != ACCIDENT_CLASS_ID:
                continue
            conf = float(box.conf[0]) * 100
            if conf > ACCIDENT_CONF_MIN:
                detected = True
                best_conf = max(best_conf, conf)

    return detected, best_conf


def generate_frames():
    """Continuously capture frames, run inference, and yield MJPEG chunks."""
    while True:
        success, frame = camera.read()
        if not success:
            break


        results = model(frame, conf=CONFIDENCE_THRESHOLD)
        detected, conf = detect_accident(results)

        annotated = results[-1].plot()

        if detected:
            state["status"] = "ACCIDENT DETECTED"
            state["confidence"] = conf
            if not state["alert_sent"]:
                log_accident(conf)
                state["alert_sent"] = True
        else:
            state["status"] = "Normal"
            state["confidence"] = 0.0
            state["alert_sent"] = False

        
        _, buffer = cv2.imencode(".jpg", annotated)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/live")
def live():
    return render_template("live.html")


@app.route("/dashboard")
def dashboard():
    accidents = (
        pd.read_csv(LOG_FILE).to_dict(orient="records")
        if os.path.exists(LOG_FILE)
        else []
    )
    return render_template("dashboard.html", accidents=accidents)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/live_status")
def live_status():
    return jsonify({
        "status": state["status"],
        "confidence": round(state["confidence"]),
    })


if __name__ == "__main__":
    app.run(debug=True)