import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import tensorflow_hub as hub
from tensorflow.keras.applications import VGG16  # or any model you need
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


#############################################
# 1. Face Detection using OpenCV Haar Cascade
#############################################
def pretrained_face_detection(x):
    """
    Detects a face using OpenCV's Haar Cascade, crops the face region,
    and resizes it to (64, 64).

    Parameters:
      x: A float32 NumPy array of shape (height, width, channels) with values in [0,1].

    Returns:
      A float32 NumPy array of shape (64, 64, channels) with values in [0,1].
    """
    img_uint8 = (x * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        resized = cv2.resize(img_uint8, (64, 64))
        return resized.astype(np.float32) / 255.0
    (x_face, y_face, w, h) = max(faces, key=lambda r: r[2] * r[3])
    face_crop = img_uint8[y_face:y_face+h, x_face:x_face+w]
    face_resized = cv2.resize(face_crop, (64, 64))
    return face_resized.astype(np.float32) / 255.0

#############################################
# 2. Face Embedding using VGGFace (ResNet50 variant)
#############################################
# Use keras_vggface instead of tfhub's FaceNet.
_face_embedding_model = None

def pretrained_face_embedding(x):
    """
    Uses ResNet50 from tf.keras.applications as a face embedding model.
    Expects input x as a float32 NumPy array with values in [0,1].
    
    Returns:
      A flattened embedding vector.
    """
    global _face_embedding_model
    if _face_embedding_model is None:
        # Create a ResNet50 model without the top classification layers,
        # using global average pooling to produce a fixed-length feature vector.
        _face_embedding_model = ResNet50(include_top=False, pooling='avg', weights='imagenet')
    # Convert input from [0,1] float to uint8 [0,255] and resize to (224,224)
    img_uint8 = (x * 255).astype(np.uint8)
    img_resized = cv2.resize(img_uint8, (224, 224)).astype(np.float32)
    # Expand dims to create a batch of size 1.
    x_input = np.expand_dims(img_resized, axis=0)
    # Preprocess the image for ResNet50.
    x_preprocessed = preprocess_input(x_input)
    # Obtain the embedding.
    embedding = _face_embedding_model.predict(x_preprocessed)
    return embedding.flatten()

#############################################
# 3. Object Detection using YOLOv3 via OpenCV DNN
#############################################
_object_detection_net = None
def pretrained_object_detection(x):
    """
    Uses YOLOv3 to detect objects in the image.
    Returns the cropped region of the highest-confidence detection, resized to (64,64).
    Expects input x as a float32 NumPy array with values in [0,1].
    """
    global _object_detection_net
    if _object_detection_net is None:
        # Ensure that "yolov3.cfg" and "yolov3.weights" are in the working directory.
        _object_detection_net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
    img_uint8 = (x * 255).astype(np.uint8)
    h, w, _ = img_uint8.shape
    blob = cv2.dnn.blobFromImage(img_uint8, scalefactor=1/255.0, size=(416,416), swapRB=True, crop=False)
    _object_detection_net.setInput(blob)
    layer_names = _object_detection_net.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in _object_detection_net.getUnconnectedOutLayers()]
    outs = _object_detection_net.forward(output_layers)
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
                x_box = int(center_x - box_w/2)
                y_box = int(center_y - box_h/2)
                boxes.append([x_box, y_box, box_w, box_h])
                confidences.append(float(confidence))
    if len(boxes) == 0:
        resized = cv2.resize(img_uint8, (64,64))
        return resized.astype(np.float32)/255.0
    max_idx = np.argmax(confidences)
    x_box, y_box, box_w, box_h = boxes[max_idx]
    x_box = max(0, x_box)
    y_box = max(0, y_box)
    x_box2 = min(w, x_box+box_w)
    y_box2 = min(h, y_box+box_h)
    crop = img_uint8[y_box:y_box2, x_box:x_box2]
    if crop.size == 0:
        resized = cv2.resize(img_uint8, (64,64))
        return resized.astype(np.float32)/255.0
    crop_resized = cv2.resize(crop, (64,64))
    return crop_resized.astype(np.float32)/255.0

#############################################
# 4. Image Segmentation using DeepLabv3 from TensorFlow Hub
#############################################
_deeplab_model = None
def pretrained_image_segmentation(x):
    """
    Uses DeepLabv3 from TensorFlow Hub to perform image segmentation.
    Returns a masked image (only the largest segment is kept) resized to (64,64).
    Expects input x as a float32 NumPy array with values in [0,1].
    """
    global _deeplab_model
    if _deeplab_model is None:
        # Append the query parameter to ensure proper loading.
        _deeplab_model = hub.load("https://tfhub.dev/tensorflow/deeplabv3/1?tf-hub-format=compressed")
    input_tensor = tf.convert_to_tensor(x[None, ...])
    result = _deeplab_model(input_tensor)
    seg_map = tf.argmax(result['default'], axis=-1)[0].numpy().astype(np.uint8)
    unique, counts = np.unique(seg_map, return_counts=True)
    if 0 in unique:
        idx = np.where(unique != 0)[0]
        if len(idx) > 0:
            unique = unique[idx]
            counts = counts[idx]
    if len(unique) == 0:
        mask = np.ones_like(seg_map, dtype=np.uint8)
    else:
        main_class = unique[np.argmax(counts)]
        mask = (seg_map == main_class).astype(np.uint8)
    img_uint8 = (x * 255).astype(np.uint8)
    segmented = cv2.bitwise_and(img_uint8, img_uint8, mask=mask)
    segmented_resized = cv2.resize(segmented, (64,64))
    return segmented_resized.astype(np.float32)/255.0

#############################################
# 5. Hand Landmark Detection using MediaPipe Hands
#############################################
def pretrained_hand_landmark_detection(x):
    """
    Uses MediaPipe Hands to detect hand landmarks.
    Returns a flattened vector of landmarks (63 values: 21 landmarks * 3 coordinates).
    Expects input x as a float32 NumPy array with values in [0,1].
    """
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        img_uint8 = (x * 255).astype(np.uint8)
        results = hands.process(cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            coords = []
            for lm in landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            return np.array(coords, dtype=np.float32)
        else:
            return np.zeros(63, dtype=np.float32)

#############################################
# 6. Body Landmark Detection using MediaPipe Pose
#############################################
def pretrained_body_landmark_detection(x):
    """
    Uses MediaPipe Pose to detect body landmarks.
    Returns a flattened vector of landmarks (99 values: 33 landmarks * 3 coordinates).
    Expects input x as a float32 NumPy array with values in [0,1].
    """
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        img_uint8 = (x * 255).astype(np.uint8)
        results = pose.process(cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            coords = []
            for lm in landmarks:
                coords.extend([lm.x, lm.y, lm.z])
            return np.array(coords, dtype=np.float32)
        else:
            return np.zeros(99, dtype=np.float32)

#############################################
# Pre-trained model application functions
#############################################
def apply_pretrained_model(x, pretrained_option):
    if pretrained_option == "Face Detection":
        return pretrained_face_detection(x)
    elif pretrained_option == "Face Embedding":
        return pretrained_face_embedding(x)
    elif pretrained_option == "Object Detection":
        return pretrained_object_detection(x)
    elif pretrained_option == "Image Segmentation":
        return pretrained_image_segmentation(x)
    elif pretrained_option == "Hand Landmark Detection":
        return pretrained_hand_landmark_detection(x)
    elif pretrained_option == "Body Landmark Detection":
        return pretrained_body_landmark_detection(x)
    else:
        return x

PRETRAINED_OUTPUT_TYPE = {
    "Face Detection": "image",
    "Object Detection": "image",
    "Image Segmentation": "image",
    "Face Embedding": "vector",
    "Hand Landmark Detection": "vector",
    "Body Landmark Detection": "vector",
    "None": "image"
}

def apply_stacked_pretrained_models(x, options):
    """
    Applies a stack of pre-trained model functions in sequence on input x.
    Allows one transition from 'image' to 'vector' output.
    Raises a ValueError if an inconsistent transition occurs.
    """
    if not options:
        return x

    expected_type = PRETRAINED_OUTPUT_TYPE.get(options[0], "image")
    out = x
    for opt in options:
        current_type = PRETRAINED_OUTPUT_TYPE.get(opt, "image")
        if current_type != expected_type:
            # Allow one transition from image -> vector
            if expected_type == "image" and current_type == "vector":
                expected_type = "vector"
            else:
                raise ValueError(
                    f"Inconsistent pretrained model stack: expected outputs to be '{expected_type}', "
                    f"but '{opt}' produces '{current_type}' output."
                )
        out = apply_pretrained_model(out, opt)
    return out
