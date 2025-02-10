import tensorflow as tf
import numpy as np

def pretrained_face_detection(x):
    """
    Dummy function for Face Detection.
    Crops the center region of the image and resizes to (64, 64).
    """
    h, w, _ = x.shape
    ch, cw = h // 2, w // 2
    crop = x[ch-50:ch+50, cw-50:cw+50, :]
    cropped = tf.image.resize(crop, (64, 64)).numpy()
    return cropped

def pretrained_face_embedding(x):
    """
    Dummy function for Face Embedding.
    Returns a random 128-dimensional embedding.
    """
    print("Applying Face Embedding pre-trained model.")
    return np.random.rand(128).astype(np.float32)

def pretrained_object_detection(x):
    """
    Dummy function for Object Detection.
    Crops a central portion of the image and resizes to (64, 64).
    """
    print("Applying Object Detection pre-trained model.")
    h, w, _ = x.shape
    crop = x[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9), :]
    cropped = tf.image.resize(crop, (64, 64)).numpy()
    return cropped

def pretrained_image_segmentation(x):
    """
    Dummy function for Image Segmentation.
    Returns the input image unmodified.
    """
    print("Applying Image Segmentation pre-trained model.")
    return x

def pretrained_hand_landmark_detection(x):
    """
    Dummy function for Hand Landmark Detection.
    Returns a random 42-dimensional vector.
    """
    print("Applying Hand Landmark Detection pre-trained model.")
    return np.random.rand(42).astype(np.float32)

def pretrained_body_landmark_detection(x):
    """
    Dummy function for Body Landmark Detection.
    Returns a random 66-dimensional vector.
    """
    print("Applying Body Landmark Detection pre-trained model.")
    return np.random.rand(66).astype(np.float32)

def apply_pretrained_model(x, pretrained_option):
    """
    Applies a pre-trained model function based on the given option.
    """
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

# Define the expected output type for each pre-trained model option.
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
    Raises a ValueError if any option in the stack produces an output type that is inconsistent.
    """
    if not options:
        return x

    # Determine expected output type from the first option.
    expected_type = PRETRAINED_OUTPUT_TYPE.get(options[0], "image")
    out = x
    for opt in options:
        current_type = PRETRAINED_OUTPUT_TYPE.get(opt, "image")
        if current_type != expected_type:
            raise ValueError(
                f"Inconsistent pretrained model stack: expected all outputs to be '{expected_type}', "
                f"but '{opt}' produces '{current_type}' output."
            )
        out = apply_pretrained_model(out, opt)
    return out
