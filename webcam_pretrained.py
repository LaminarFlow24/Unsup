import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, render_template, Response, request
import time

app = Flask(__name__)

#############################################
# Global initializations for MediaPipe and OpenCV
#############################################
# Face detection using OpenCV Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# MediaPipe Face Mesh for face embedding (mesh overlay)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# MediaPipe Hands for hand landmark detection
mp_hands = mp.solutions.hands

# MediaPipe Pose for body landmark detection
mp_pose = mp.solutions.pose

# MediaPipe Selfie Segmentation for segmentation demo
mp_selfie_segmentation = mp.solutions.selfie_segmentation

#############################################
# Processing Functions
#############################################
def process_face_detection(frame):
    """Detect faces using Haar Cascade and draw bounding boxes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def process_face_embedding(frame):
    """
    Uses MediaPipe Face Mesh to detect facial landmarks and overlays the face mesh.
    (This simulates a face embedding visualization by drawing landmarks.)
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
    else:
        cv2.putText(frame, "No face detected", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame

def process_object_detection(frame):
    """
    Uses YOLOv3 via OpenCV DNN to detect objects and draw the bounding box of the highest-confidence detection.
    (Ensure that "yolov3.cfg" and "yolov3.weights" exist in your working directory.)
    """
    net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
    img_uint8 = frame.copy()
    h, w, _ = img_uint8.shape
    blob = cv2.dnn.blobFromImage(img_uint8, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                box_w = int(detection[2] * w)
                box_h = int(detection[3] * h)
                x_box = int(center_x - box_w / 2)
                y_box = int(center_y - box_h / 2)
                boxes.append([x_box, y_box, box_w, box_h])
                confidences.append(float(confidence))
    if boxes:
        max_idx = np.argmax(confidences)
        x_box, y_box, box_w, box_h = boxes[max_idx]
        cv2.rectangle(frame, (x_box, y_box), (x_box+box_w, y_box+box_h), (0, 255, 0), 2)
        cv2.putText(frame, f"Obj Conf: {confidences[max_idx]:.2f}", (x_box, y_box-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame

def process_image_segmentation(frame):
    """
    Applies MediaPipe Selfie Segmentation to segment the person from the background,
    and overlays the segmentation mask.
    """
    segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = segmentation.process(rgb)
    if results.segmentation_mask is not None:
        mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
        frame = cv2.bitwise_and(frame, frame, mask=mask)
    return frame

def process_hand_landmark(frame):
    """
    Uses MediaPipe Hands to detect hand landmarks and draws them.
    """
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame

def process_body_landmark(frame):
    """
    Uses MediaPipe Pose to detect body landmarks and draws them.
    """
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return frame

#############################################
# Mapping mode to processing function
#############################################
PROCESSING_FUNCTIONS = {
    "face_detection": process_face_detection,
    "face_embedding": process_face_embedding,
    "object_detection": process_object_detection,
    "image_segmentation": process_image_segmentation,
    "hand_landmark": process_hand_landmark,
    "body_landmark": process_body_landmark
}

#############################################
# Video streaming generator function
#############################################
def generate_frames(mode):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    processing_fn = PROCESSING_FUNCTIONS.get(mode, process_face_detection)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = processing_fn(frame.copy())
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

#############################################
# Flask routes
#############################################
@app.route('/')
def index():
    # Simple HTML form to select the mode
    return '''
    <html>
      <head>
        <title>Webcam Pre-trained Demo</title>
      </head>
      <body>
        <h1>Select Pre-trained Model Mode</h1>
        <form action="/video_feed">
          <select name="mode">
            <option value="face_detection">Face Detection</option>
            <option value="face_embedding">Face Embedding (Face Mesh)</option>
            <option value="object_detection">Object Detection</option>
            <option value="image_segmentation">Image Segmentation</option>
            <option value="hand_landmark">Hand Landmark Detection</option>
            <option value="body_landmark">Body Landmark Detection</option>
          </select>
          <input type="submit" value="Start Webcam">
        </form>
      </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    mode = request.args.get("mode", "face_detection")
    return Response(generate_frames(mode),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
