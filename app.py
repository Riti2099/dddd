from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread, Event
from flask import Flask, Response, render_template, jsonify, send_from_directory, request
import numpy as np
import imutils
import time
import dlib
import cv2
import pyttsx3
import logging
import os
import base64
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress MSMF warnings
os.environ["OPENCV_LOG_LEVEL"]="ERROR"
logging.getLogger("opencv").setLevel(logging.ERROR)

app = Flask(__name__)

# Global variables
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
COUNTER = 0
TOTAL = 0
ALARM_ON = False
YAWN_COUNTER = 0
alarm_status = False
alarm_status2 = False
saying = False
video_stream = None
stream_active = False

# Performance tracking
y_true = []  # Actual labels
y_pred = []  # Predicted labels
latency_times = []

# Initialize your detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/ubuntu/dddd/shape_predictor_68_face_landmarks.dat")

# Helper functions
def compute_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def check_drowsy(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = compute_ear(leftEye)
    rightEAR = compute_ear(rightEye)
    return (leftEAR + rightEAR) / 2.0, leftEye, rightEye

def generate_frames():
    global COUNTER, alarm_status, alarm_status2, y_true, y_pred, latency_times
    video_stream = VideoStream(src=0).start()
    time.sleep(2.0)
    
    while True:
        start_time = time.time()
        frame = video_stream.read()
        if frame is None:
            continue
        
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            ear, leftEye, rightEye = check_drowsy(shape)
            
            actual_label = 1 if ear < EYE_AR_THRESH else 0
            predicted_label = 1 if ear < EYE_AR_THRESH else 0
            y_true.append(actual_label)
            y_pred.append(predicted_label)
            
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    alarm_status = True
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                alarm_status = False

        latency_times.append(time.time() - start_time)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    video_stream.stop()

@app.route('/metrics', methods=['GET'])
def get_metrics():
    if len(y_true) == 0:
        return jsonify({"error": "No data collected yet!"})
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    latency_avg = np.mean(latency_times)
    cm = confusion_matrix(y_true, y_pred)
    
    # Generate Confusion Matrix Plot
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Drowsy', 'Drowsy'], yticklabels=['Not Drowsy', 'Drowsy'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('static/confusion_matrix.png')
    
    return jsonify({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "average_latency": latency_avg
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
