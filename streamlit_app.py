import streamlit as st
import numpy as np
from PIL import Image
from keras.engine.saving import load_model
from keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
import os
import cv2
from keras.utils import normalize
from unet import simple_unet_model
import time
#from pre_processing import my_PreProc
#diabetic retinopathy classifications

def predict(model_name, img):

        results = img_to_array(img)
        arr = results.reshape(-1, 256, 256, 3)
        if model_name == "My_convolution_layer_model":
             new_model = load_model('C:/Users/16189\Documents/hyperspectral/retina_augmented_model__new_mymodel.h5')
        elif model_name == "Transfer learning":
            new_model = load_model('C:/Users/16189\Documents/hyperspectral/retina_augmented_model__new_mymodel_transfer.h5')
        # arr = results.reshape((1,)+results.shape)
        results = new_model.predict(arr)

        # Now predict using the trained RF model.
        # prediction_RF = model1.predict(X_test_features)
        if results == 0:
            x = "diabetic"
        else:
            x = "no_diabetic"

        return x



def get_model():
    return simple_unet_model(patch_size, patch_size, 1)


##Exudates segmentation
def prediction(model, image, patch_size):
    segm_img = np.zeros(image.shape[:2])  # Array with zeros to be filled with segmented values
    patch_num = 1
    my_bar = st.progress(0)
    for i in range(0, image.shape[0], patch_size):  # Steps of 256
        for j in range(0, image.shape[1], patch_size):  # Steps of 256
            # print(i, j)
            single_patch = image[i:i + patch_size, j:j + patch_size]
            single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1), 2)
            single_patch_shape = single_patch_norm.shape[:2]
            single_patch_input = np.expand_dims(single_patch_norm, 0)
            single_patch_prediction = (model.predict(single_patch_input)[0, :, :, 0] > 0.5).astype(np.uint8)
            segm_img[i:i + single_patch_shape[0], j:j + single_patch_shape[1]] += cv2.resize(single_patch_prediction,
                                                                                             single_patch_shape[::-1])

            # print("Finished processing patch number ", patch_num, " at position ", i, j)
            patch_num += 1
            # st.write(patch_num)
            time.sleep(0.1)
            # while patch_num <= 100:
            #     my_bar.progress(patch_num + 1)
    return segm_img


def predictions(file, model, patch_size):
    large_image = cv2.imread(file)
    large_image=cv2.cvtColor(large_image,cv2.COLOR_BGR2GRAY)
    large_image = cv2.resize(large_image, (2048, 2048))

    segmented_image = prediction(model, large_image, patch_size)

    return segmented_image


def get_image(model_name, name):
    if name == "Diabetic retinopathy classification":
        image = st.sidebar.file_uploader(label="Select an retinal fundus image", type=['jpg', 'jpeg', 'png'])
        if image is not None:
            image_data = image.read()
            # uploaded_img=load_img(image_data, target_size=(256, 256))
            uploaded_img2 = Image.open(BytesIO(image_data))
            uploaded_img2 = uploaded_img2.resize((256, 256))
            st.write(image.name)

            # uploaded_img3=cv2.resize(uploaded_img2,(256,256))
            st.image(uploaded_img2)
            button = st.button("Click to predict the classification results")
            if button:
                prediction = predict(model_name, uploaded_img2)
                st.write("# The Prediction is:")
                st.write(prediction)
            # predict
    else:
        image = st.sidebar.file_uploader(label="Select an retinal funcdus image", type=['jpg', 'jpeg', 'png'])
        # st.warning("The image should be divisible by the path size")
        if image is not None:
            image_data = image.read()
            # uploaded_img=load_img(image_data, target_size=(256, 256))
            uploaded_img2 = Image.open(BytesIO(image_data))
            st.write(image.name)
            st.image(uploaded_img2)
            new_path = 'C:/Users/16189/Documents/hyperspectral/directory'
            # saving image file
            with open(os.path.join(new_path, image.name, ), "wb") as f:
                f.write(image.getbuffer())
            st.success('File saved')
            button=st.button("Click to predict the segmentation  results")
            if button:
                image_path = 'C:/Users/16189/Documents/hyperspectral/directory/' + image.name
                model = get_model()

                model.load_weights('C:/Users/16189/Documents/hyperspectral/retinal_exudates_segmentation2.h5')
                import time

                st.header("Segmented Image")
                segmented_image = predictions(image_path, model, patch_size)
                with open(os.path.join(new_path, "segmented"+image.name, ), "wb") as f:
                    f.write(image.getbuffer())

                # Large image
                st.balloons()
                st.image(segmented_image)




nav_bar = st.sidebar.radio("Navigation", ["Home", "About"])
if nav_bar == "Home":

    st.title("This is the app of Detecting Diabetic Retinopathy and segmention of the exudates ")
    st.markdown("***")
    st.write("""
    # Model
    Which model you want to work on?
    """)
    name = st.sidebar.selectbox("Select Dataset", ("Diabetic retinopathy classification", "segmentation"))

    st.text(name)
    st.write("Model")

    if name == "Diabetic retinopathy classification":
        model_name = st.sidebar.selectbox("Select the model",
                                          ("Transfer learning", "My_convolution_layer_model", "Eye net"))
    else:
        model_name = st.sidebar.selectbox("Select the model", ("Unet", "Else"))
        patch_size = st.sidebar.selectbox("Select the patch size for segmentation", (256, 512,1024))

    st.text(model_name)
    get_image(model_name, name)
if nav_bar=="About":
    st.header(" Hi, I am Hridoy Biswas and the main developer of the apps.")

    st.markdown("***")
    st.text(" # Ihis is the app for the classification of the diabetic retinopathy."
            "# I used three deep learning model and developed one of my model.You can use all three models and predict ")
    st.text(" The segmentation model is u-net model.YOu can segment the exudates")
