import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

st.title("Traffic Sign Recognition")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    model = load_model('enhanced_model.h5')
    prediction = model.predict(img_array)
    st.write(f"Predicted Traffic Sign Category: {np.argmax(prediction)}")
