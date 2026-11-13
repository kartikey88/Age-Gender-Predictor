# Age & Gender predictor
# Run with:  streamlit run app.py
# Needs age_gender_model_ft.keras in the same folder as this file.

import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.applications.resnet50 import preprocess_input

MODEL_FILE = "age_gender_model_ft.keras"
GENDERS = ["Male", "Female"]   # the dataset uses 0 = male, 1 = female

st.title("Age & Gender Predictor")
st.write("Upload a photo or take one with your camera. "
         "It works best with a clear, front-facing face.")

# OpenCV 5 dropped the face detector we use, so check for it first
if not hasattr(cv2, "CascadeClassifier"):
    st.error('This OpenCV version has no face detector. Run: pip install "opencv-python<5"')
    st.stop()


@st.cache_resource
def load_model():
    folder = os.path.dirname(os.path.abspath(__file__))
    return tf.keras.models.load_model(os.path.join(folder, MODEL_FILE), compile=False)


@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def predict(face):
    """Takes an RGB face crop, returns (age, gender, confidence)."""
    face = cv2.resize(face, (200, 200)).astype("float32")
    batch = preprocess_input(face[np.newaxis])    # same preprocessing as training
    age_out, gender_out = model(batch, training=False)

    age = int(round(float(age_out[0][0])))
    female_prob = float(gender_out[0][0])
    if female_prob >= 0.5:
        return age, "Female", female_prob
    return age, "Male", 1 - female_prob


folder = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(folder, MODEL_FILE)):
    st.error(f"Can't find {MODEL_FILE}. Put it in the same folder as app.py.")
    st.stop()

model = load_model()
detector = load_face_detector()

choice = st.radio("Input", ["Upload a photo", "Use camera"], horizontal=True)
if choice == "Upload a photo":
    picture = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])
else:
    picture = st.camera_input("Take a photo")

if picture is not None:
    # exif_transpose fixes photos from phones that come out sideways
    image = ImageOps.exif_transpose(Image.open(picture)).convert("RGB")
    frame = np.array(image)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 6, minSize=(60, 60))

    if len(faces) == 0:
        st.image(image)
        st.warning("I couldn't find a face. Try a clearer, front-facing photo.")

    results = []
    for (x, y, w, h) in faces:
        pad = int(w * 0.15)    # a bit of extra space around the face
        crop = frame[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
        age, gender, confidence = predict(crop)
        results.append((age, gender, confidence))

        # draw the box and a label (black outline + green text so it's easy to read)
        text = f"{gender}, ~{age}"
        scale = max(0.6, frame.shape[1] / 800)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(frame, text, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 200, 0), 2, cv2.LINE_AA)

    if results:
        st.image(frame)

    for number, (age, gender, confidence) in enumerate(results, start=1):
        if len(results) > 1:
            st.subheader(f"Face {number}")
        left, right = st.columns(2)
        left.metric("Estimated age", f"~{age} years")
        right.metric("Estimated gender", gender)
        st.progress(confidence, text=f"Gender confidence: {confidence:.0%}")

st.caption("Age is only an estimate and can be off by several years. "
           "This is a demo, not something to base decisions on.")