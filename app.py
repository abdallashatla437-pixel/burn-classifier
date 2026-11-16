import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf

BASE_DIR = Path(__file__).parent
CLASS_NAMES = ["first_degree", "second_degree", "third_degree"]  # same order as training
IMG_SIZE = (224, 224)   # change if you trained with another size
NORMALIZE = True        # set False if your model already has a Rescaling layer

@st.cache_resource
def load():
    files = list(BASE_DIR.glob("*.keras")) + list(BASE_DIR.glob("*.h5"))
    if not files:
        st.error("No model file (.keras or .h5) found next to app.py. Upload it to GitHub.")
        st.stop()
    model = tf.keras.models.load_model(files[0])
    return model, CLASS_NAMES

model, class_names = load()

st.title("Burn Classifier")
uploaded = st.file_uploader("Upload a burn image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_container_width=True)

    x = np.array(img.resize(IMG_SIZE), dtype=np.float32)
    if NORMALIZE:
        x = x / 255.0
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x)[0]
    best = int(np.argmax(probs))

    st.subheader(f"Prediction: {class_names[best]}")
    st.write(f"Confidence: {probs[best] * 100:.1f}%")
    for name, p in zip(class_names, probs):
        st.progress(float(p), text=f"{name}: {p * 100:.1f}%")
