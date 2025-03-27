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

# Initialize detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/ubuntu/dddd/shape_predictor_68_face_landmarks.dat")

def compute_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def generate_frames():
    global video_stream, COUNTER, stream_active, alarm_status, alarm_status2
    
    try:
        if video_stream is None:
            try:
                video_stream = VideoStream(src=0).start()
                time.sleep(2.0)
                stream_active = True
            except Exception as e:
                print(f"Error initializing VideoStream: {e}")
                return

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

                # Draw facial features
                cv2.polylines(frame, [shape[0:17]], False, (0, 255, 255), 1)  # Jawline
                cv2.polylines(frame, [shape[27:36]], True, (255, 255, 0), 1)  # Nose
                cv2.drawContours(frame, [cv2.convexHull(shape[48:60])], -1, (255, 0, 0), 1)  # Mouth
                
                ear = (compute_ear(shape[36:42]) + compute_ear(shape[42:48])) / 2.0
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        alarm_status = True
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    COUNTER = 0
                    alarm_status = False
                
                cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
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
