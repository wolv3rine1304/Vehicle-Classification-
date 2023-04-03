import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

model = load_model('20BCP129.h5')
CLASS_NAMES = ["non-vehicles", "vehicles"]

st.title('Image classification for vehicles')
st.markdown('Upload Image')

dog_image = st.file_uploader("Upload image")
submit = st.button('Predict')
if submit:
    if dog_image is not None:
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        st.image(opencv_image, channels="BGR")
        opencv_image = cv2.resize(opencv_image, (50, 50))
        opencv_image.shape = (1, 50, 50, 3)
        Y_pred = model.predict(opencv_image)
        ypred1 = np.round(Y_pred)
        ypred1 = np.asarray(ypred1, dtype='int')
        predict = ""
        if ypred1[0] == 0:
            predict = "Not Vehicle"
        else:
            predict = "Vehicle"
        st.title(predict)
