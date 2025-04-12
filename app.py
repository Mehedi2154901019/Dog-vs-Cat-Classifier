import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Constants
IMG_SIZE = 150
CLASS_NAMES = ['cats', 'dogs']  # Assuming binary classification: 0 = cat, 1 = dog

# Load the saved model
@st.cache_resource
def load_dogcat_model():
    return load_model("dogcat.keras")

model = load_dogcat_model()

# App Title
st.title("🐶🐱 Dog vs Cat Classifier")
st.write("Upload an image, and the model will tell you whether it's a Dog or a Cat.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image in a clean size to avoid scroll issues
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)  # Cleaner preview

    # Predict button
    if st.button("Predict"):
        # Preprocess the image
        img_resized = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)[0][0]
        predicted_label = CLASS_NAMES[1] if prediction > 0.5 else CLASS_NAMES[0]
        confidence = prediction if prediction > 0.5 else 1 - prediction

        # Display prediction
        st.markdown(f"### Prediction: `{predicted_label.upper()}`")
        st.markdown(f"### Confidence: `{confidence * 100:.2f}%`")
