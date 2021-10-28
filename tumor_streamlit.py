import streamlit as st
import numpy as np
from tensorflow.keras.models import Model, load_model
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import cv2
import os
import imutils
import numpy as np
import matplotlib.pyplot as plt
import time
from os import listdir
from tensorflow.keras.preprocessing import image

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def tumour(url): 
    st.markdown(f'<p style="color:#ff0000;font-size:35px;"><strong>{url}</strong></p>', unsafe_allow_html=True)
    
def non_tumour(url): 
    st.markdown(f'<p style="color:#32CD32;font-size:35px;"><strong>{url}</strong></p>', unsafe_allow_html=True)
    
def crop_brain_contour(image, plot=False):
    
    #import imutils
    #import cv2
    #from matplotlib import pyplot as plt
    
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            

    if plot:
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        
        plt.tick_params(axis='both', which='both', 
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        
        plt.title('Original Image')
            
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)

        plt.tick_params(axis='both', which='both', 
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Cropped Image')
        
        plt.show()
    
    return new_image

def predict():

    col1, col2 = st.columns([1,3])

    with col1:
         st.image("woxsen_logo.png")


    with col2:
        st.title("Brain Tumour Prediction")

    
        


    uploaded_file = st.file_uploader("Upload image for prediction", type = ["jfif","jpeg","png","jpg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
            
        image.save("test_{}".format(uploaded_file.name))
        path = os.path.abspath(os.getcwd()) + "\\test_{}".format(uploaded_file.name)
        
        image_width, image_height = (240, 240)
        image = cv2.imread(path)
        image = crop_brain_contour(image, plot=False)
        image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
        image = image / 255.
        image = image.reshape(1,240,240,3)
        
        best_model = load_model(filepath='models/cnn-parameters-improvement-23-0.91.model')

        output = best_model.predict(image)

        col1,col2, col3 = st.columns([3,0.2,2])

        with col1:
            st.write("Original Image")
            image = cv2.imread(path)
            st.image(image)

        with col3:
            st.write("Diagonised Image")
            image = cv2.imread(path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, 0.7)
            (T, thresh) = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
            (T, threshInv) = cv2.threshold(gray, 155, 255,cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            closed = cv2.erode(closed, None, iterations = 14)
            closed = cv2.dilate(closed, None, iterations = 13)
            edged = cv2.Canny(closed,100,200)
            (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, cnts, -1, (255, 0, 0), 2)

            st.image(image)
            
        st.write("\n\n")
       
        if output[0][0] < 0.5:
            col1,col2,col3 = st.columns([1,1,6])

            with col3:
                non_tumour("It is Non-Tumorous")
        else:
            col1,col2,col3 = st.columns([1,1,6])

            with col3:
                tumour("Tumour has been detected")
                #tumour("The given sample is Tumorous")

        
if __name__ == "__main__":
    predict()

