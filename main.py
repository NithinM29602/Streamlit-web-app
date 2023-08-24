import streamlit as st

st.header("Welcome to my World")

import cv2
import streamlit as st
import numpy as np

cap = cv2.VideoCapture(0)

st.title("Webcam Video Stream")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform your processing on 'frame' here

    # Display the frame in Streamlit
    st.image(frame, channels="BGR", use_column_width=True)

cap.release()
