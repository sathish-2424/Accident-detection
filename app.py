from flask import Flask, render_template, Response, jsonify
import cv2, csv, os
from ultralytics import YOLO
from datetime import datetime
import pandas as pd

app = Flask(__name__)
model = YOLO("best.pt")
camera = cv2.VideoCapture(0)

accident_status = "Normal"
confidence_score = 0
alert_sent = False

LOG_FILE = "accident_log.csv"

def log_accident(conf):
    exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["Date", "Time", "Confidence"])
        now = datetime.now()
        writer.writerow([now.date(), now.time(), conf])

def generate_frames():
    global accident_status, confidence_score, alert_sent

    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame, conf=0.5)
        detected = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                confidence_score = float(box.conf[0]) * 100
                if cls == 0 and confidence_score > 70:
                    detected = True
            frame = r.plot()

        if detected:
            accident_status = "ACCIDENT DETECTED"
            if not alert_sent:
                log_accident(confidence_score)
                alert_sent = True
        else:
            accident_status = "Normal"
            alert_sent = False

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/live")
def live():
    return render_template("live.html")

@app.route("/dashboard")
def dashboard():
    if os.path.exists(LOG_FILE):
        data = pd.read_csv(LOG_FILE).to_dict(orient="records")
    else:
        data = []
    return render_template("dashboard.html", accidents=data)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/live_status")
def live_status():
    return jsonify({"status": accident_status, "confidence": int(confidence_score)})

if __name__ == "__main__":
    app.run(debug=True)
