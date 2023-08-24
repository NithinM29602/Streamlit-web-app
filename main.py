import streamlit as st
from streamlit_camera_input_live import camera_input_live

frame = camera_input_live()

# Display the frame using Streamlit
st.image(frame)

cap.release()
