# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image, ImageOps

# Load the trained model
model = tf.keras.models.load_model("vgg16_traffic.h5")

# Class labels
class_descriptions = {
    "0":  "Speed limit (20 km/h)",
    "1":  "Speed limit (30 km/h)",
    "2":  "Speed limit (50 km/h)",
    "3":  "Speed limit (60 km/h)",
    "4":  "Speed limit (70 km/h)",
    "5":  "Speed limit (80 km/h)",
    "6":  "End of speed limit (80 km/h)",
    "7":  "Speed limit (100 km/h)",
    "8":  "Speed limit (120 km/h)",
    "9":  "No passing",
    # 👉 Add more up to "57"
}

st.title("🚦 Traffic Sign Recognition using VGG16")

uploaded_file = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_obj = Image.open(uploaded_file).convert('RGB')
    st.image(image_obj, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = ImageOps.fit(image_obj, (224, 224), Image.ANTIALIAS)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    label = str(class_idx)
    description = class_descriptions.get(label, "Unknown sign")

    st.success(f"✅ Prediction: {description}")
    st.info(f"📊 Confidence: {confidence:.2f}%")
