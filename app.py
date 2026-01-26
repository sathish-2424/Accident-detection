from flask import Flask, render_template, Response, jsonify
import cv2, csv, os
from ultralytics import YOLO
from datetime import datetime
import pandas as pd
import threading
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load model
model = YOLO("best.pt")

# Configuration - Replace with your IP camera URL
CAMERA_URL = os.getenv("CAMERA_URL", "rtsp://192.168.1.100:554/stream")  # Change this

# Global variables
accident_status = "Normal"
confidence_score = 0
alert_sent = False
current_frame = None
camera = None

LOG_FILE = "accident_log.csv"

def log_accident(conf):
    """Log accident to CSV file"""
    exists = os.path.isfile(LOG_FILE)
    try:
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["Date", "Time", "Confidence"])
            now = datetime.now()
            writer.writerow([now.date(), now.time(), conf])
        logging.info(f"Accident logged with confidence: {conf}")
    except Exception as e:
        logging.error(f"Error logging accident: {e}")

def connect_camera():
    """Connect to IP camera with retry logic"""
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            cap = cv2.VideoCapture(CAMERA_URL)
            if cap.isOpened():
                logging.info("Camera connected successfully")
                return cap
            else:
                logging.warning(f"Failed to open camera, retry {retry_count + 1}/{max_retries}")
                retry_count += 1
        except Exception as e:
            logging.error(f"Camera connection error: {e}")
            retry_count += 1
    
    logging.error("Failed to connect to camera after retries")
    return None

def generate_frames():
    """Generate video frames with accident detection"""
    global accident_status, confidence_score, alert_sent, current_frame, camera
    
    camera = connect_camera()
    if camera is None:
        logging.error("No camera available")
        return

    frame_count = 0
    
    while True:
        try:
            success, frame = camera.read()
            
            if not success:
                logging.warning("Failed to read frame, attempting reconnect...")
                camera.release()
                camera = connect_camera()
                if camera is None:
                    break
                continue
            
            # Resize frame for faster processing (optional)
            frame = cv2.resize(frame, (640, 480))
            current_frame = frame.copy()
            
            # Run YOLO detection
            results = model(frame, conf=0.5)
            detected = False
            
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    confidence_score = float(box.conf[0]) * 100
                    
                    # Class 0 = accident (adjust based on your model)
                    if cls == 0 and confidence_score > 70:
                        detected = True
                
                # Draw results on frame
                frame = r.plot()
            
            # Update accident status
            if detected:
                accident_status = "ACCIDENT DETECTED"
                if not alert_sent:
                    log_accident(confidence_score)
                    alert_sent = True
            else:
                accident_status = "Normal"
                alert_sent = False
            
            # Encode frame to JPEG
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            
            frame_count += 1
            if frame_count % 30 == 0:
                logging.info(f"Processed {frame_count} frames")
        
        except Exception as e:
            logging.error(f"Error in frame generation: {e}")
            break
    
    if camera:
        camera.release()

@app.route("/")
def index():
    """Home page"""
    return render_template("index.html")

@app.route("/live")
def live():
    """Live streaming page"""
    return render_template("live.html")

@app.route("/dashboard")
def dashboard():
    """Dashboard with accident logs"""
    try:
        if os.path.exists(LOG_FILE):
            data = pd.read_csv(LOG_FILE).to_dict(orient="records")
        else:
            data = []
        return render_template("dashboard.html", accidents=data)
    except Exception as e:
        logging.error(f"Error loading dashboard: {e}")
        return render_template("dashboard.html", accidents=[])

@app.route("/video_feed")
def video_feed():
    """Video streaming endpoint"""
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/live_status")
def live_status():
    """Get current accident status and confidence"""
    return jsonify({
        "status": accident_status,
        "confidence": int(confidence_score),
        "timestamp": datetime.now().isoformat()
    })

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route("/logs")
def get_logs():
    """Get all accident logs as JSON"""
    try:
        if os.path.exists(LOG_FILE):
            data = pd.read_csv(LOG_FILE).to_dict(orient="records")
            return jsonify(data), 200
        return jsonify([]), 200
    except Exception as e:
        logging.error(f"Error retrieving logs: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)