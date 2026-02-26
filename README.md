# Accident Detection System

An intelligent accident detection system built with Flask and YOLOv8 that uses computer vision to detect accidents in real-time video feeds and logs them with timestamps and confidence scores.

## Features

- **Real-time Detection**: Detects accidents using a trained YOLOv8 model
- **Live Dashboard**: Web-based interface to monitor accident detection status
- **Logging System**: Automatically logs detected accidents with date, time, and confidence scores
- **Confidence Scoring**: Reports confidence levels for each detection
- **Web Interface**: Easy-to-use dashboard for monitoring and viewing logs

## Project Structure

```
├── app.py                 # Main Flask application
├── best.pt               # Trained YOLOv8 model weights
├── accident_log.csv      # Log file for detected accidents
├── requirements.txt      # Python dependencies
├── static/
│   └── style.css        # CSS styling
└── templates/
    ├── index.html       # Home page
    ├── dashboard.html   # Main dashboard
    └── live.html        # Live feed viewer
```

## Installation

1. Clone or download this repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have a webcam connected to your system (or modify the code to use a different video source)

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. The system will:
   - Stream live video from your camera
   - Detect accidents in real-time
   - Display detection status and confidence scores
   - Log all accidents with timestamps to `accident_log.csv`

## Dependencies

- **Flask**: Web framework for the dashboard
- **OpenCV**: Computer vision library for video processing
- **YOLOv8 (Ultralytics)**: Object detection model
- **Pandas**: Data analysis and CSV handling

## How It Works

1. The application captures video frames from the connected camera
2. Each frame is processed through the YOLOv8 model trained on accident detection
3. When an accident is detected with confidence > 70%, it:
   - Updates the dashboard status
   - Logs the detection to CSV
   - Sends an alert
4. The results are displayed in real-time on the web interface

## CSV Log Format

The `accident_log.csv` file contains the following columns:
- **Date**: Date when accident was detected (YYYY-MM-DD)
- **Time**: Time when accident was detected (HH:MM:SS)
- **Confidence**: Confidence score of the detection (0-100)

## Configuration

You can modify detection parameters in `app.py`:
- Confidence threshold: Change `conf=0.5` in the model prediction
- Detection threshold: Adjust `confidence_score > 70` for sensitivity
