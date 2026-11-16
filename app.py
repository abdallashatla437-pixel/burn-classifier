import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Burn Type Classifier", page_icon="🩹")

@st.cache_resource
def load():
    model = tf.keras.models.load_model("burn_model.keras")
    names = json.load(open("class_names.json"))
    return model, names

model, class_names = load()

NICE = {
    "first_degree": "First-degree burn",
    "second_degree": "Second-degree burn",
    "third_degree": "Third-degree burn",
}

st.title("Burn Type Classifier")
st.caption("Student project. Not medical advice. See a doctor for any real burn.")

file = st.file_uploader("Upload a photo of the burn", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, width=300)

    arr = np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), axis=0)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))

    st.subheader(f"Result: {NICE.get(class_names[idx], class_names[idx])}")
    for n, p in zip(class_names, probs):
        st.write(f"{NICE.get(n, n)}: {p*100:.1f}%")
        st.progress(float(p))
