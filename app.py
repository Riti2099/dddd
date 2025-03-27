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

# Initialize your detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/ubuntu/dddd/shape_predictor_68_face_landmarks.dat")

def compute_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def check_drowsy(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = compute_ear(leftEye)
    rightEAR = compute_ear(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def check_yawn(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

def generate_frames():
    global video_stream, COUNTER, stream_active, alarm_status, alarm_status2
    
    try:
        if video_stream is None:
            video_stream = VideoStream(src=0).start()
            time.sleep(2.0)
            stream_active = True

        while stream_active:
            frame = video_stream.read()
            if frame is None:
                continue

            frame = imutils.resize(frame, width=450)
            height, width, _ = frame.shape
            print(f"Resolution: {width}x{height}")  # Print resolution to console
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                
                # Draw eye contours
                (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
                (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                
                # Draw mouth contour
                mouth = shape[48:60]
                cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 0, 0), 1)
                
                # Draw jawline
                jaw = shape[0:17]
                cv2.polylines(frame, [jaw], False, (0, 255, 255), 1)
                
                # Draw nose contour
                nose = shape[27:36]
                cv2.polylines(frame, [nose], True, (255, 255, 0), 1)
                
                ear, _, _ = check_drowsy(shape)
                distance = check_yawn(shape)
                
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        alarm_status = True
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    COUNTER = 0
                    alarm_status = False
                
                if distance > YAWN_THRESH:
                    alarm_status2 = True
                    cv2.putText(frame, "Yawn Alert", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    alarm_status2 = False
                
                cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"YAWN: {distance:.2f}", (300, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        if video_stream is not None:
            video_stream.stop()
            video_stream = None
        stream_active = False

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
