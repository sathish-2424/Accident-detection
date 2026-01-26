from flask import Flask, render_template, Response, jsonify
import cv2
import csv
import os
from ultralytics import YOLO
from datetime import datetime
import pandas as pd
import logging
import threading

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YOLO model
try:
    model = YOLO("best.pt")
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    model = None

# Configuration
CAMERA_URL = os.getenv("CAMERA_URL", "0")
LOG_FILE = "accident_log.csv"

# Global variables
accident_status = "Normal"
confidence_score = 0
alert_sent = False
current_frame = None
camera = None
camera_lock = threading.Lock()

def log_accident(conf):
    """Log accident detection to CSV file"""
    exists = os.path.isfile(LOG_FILE)
    try:
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["Date", "Time", "Confidence"])
            now = datetime.now()
            writer.writerow([now.date(), now.time(), conf])
        logger.info(f"Accident logged with confidence: {conf}")
    except Exception as e:
        logger.error(f"Error logging accident: {e}")

def connect_camera():
    """Connect to camera with retry logic"""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            logger.info(f"Attempting to connect to camera: {CAMERA_URL}")
            cap = cv2.VideoCapture(CAMERA_URL)
            
            # Set timeout
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if cap.isOpened():
                logger.info("Camera connected successfully")
                return cap
            else:
                logger.warning(f"Failed to open camera, retry {retry_count + 1}/{max_retries}")
                cap.release()
                retry_count += 1
        except Exception as e:
            logger.error(f"Camera connection error: {e}")
            retry_count += 1
    
    logger.error("Failed to connect to camera after retries")
    return None

def generate_frames():
    """Generate video frames with accident detection"""
    global accident_status, confidence_score, alert_sent, current_frame, camera, model
    
    camera = connect_camera()
    if camera is None:
        logger.error("No camera available")
        return

    frame_count = 0
    
    while True:
        try:
            success, frame = camera.read()
            
            if not success:
                logger.warning("Failed to read frame, attempting reconnect...")
                camera.release()
                camera = connect_camera()
                if camera is None:
                    break
                continue
            
            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))
            current_frame = frame.copy()
            
            # Run YOLO detection
            if model is not None:
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
                logger.info(f"Processed {frame_count} frames - Status: {accident_status}")
        
        except Exception as e:
            logger.error(f"Error in frame generation: {e}")
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
        logger.error(f"Error loading dashboard: {e}")
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
        "timestamp": datetime.now().isoformat(),
        "camera_url": CAMERA_URL
    })

@app.route("/health")
def health():
    """Health check endpoint for Render"""
    return jsonify({
        "status": "healthy",
        "camera": "connected" if camera and camera.isOpened() else "disconnected",
        "model": "loaded" if model is not None else "not loaded"
    }), 200

@app.route("/logs")
def get_logs():
    """Get all accident logs as JSON"""
    try:
        if os.path.exists(LOG_FILE):
            data = pd.read_csv(LOG_FILE).to_dict(orient="records")
            return jsonify(data), 200
        return jsonify([]), 200
    except Exception as e:
        logger.error(f"Error retrieving logs: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/reset_logs", methods=["POST"])
def reset_logs():
    """Reset accident logs"""
    try:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
            logger.info("Accident logs reset")
            return jsonify({"message": "Logs reset successfully"}), 200
        return jsonify({"message": "No logs to reset"}), 200
    except Exception as e:
        logger.error(f"Error resetting logs: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)