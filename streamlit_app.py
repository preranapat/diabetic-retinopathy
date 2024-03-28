import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Function to load and preprocess the image
def preprocess_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

# Function to load the pre-trained model
def load_model():
    model = tf.keras.models.load_model("path_to_your_pretrained_model.h5")
    return model

# Function to make predictions
def predict(image, model):
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

# Streamlit app
def main():
    st.title("Diabetic Retinopathy Detection")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Load the pre-trained model
        model = load_model()

        # Make predictions
        prediction = predict(image, model)
        
        # Displaying results
        if prediction[0][0] > 0.5:
            st.write("Prediction: Diabetic Retinopathy")
        else:
            st.write("Prediction: Healthy")

if __name__ == "__main__":
    main()
