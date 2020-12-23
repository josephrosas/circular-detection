# Making the necessary imports
import cv2
import math
import statistics
import numpy as np
import streamlit as st

from PIL import Image

st.title("Detect Circular-Shaped Objects")
st.markdown("This simple web application detects circular-shaped objects from uploaded images.")

st.sidebar.title("Image Adjustments")
st.sidebar.markdown("Perform any necessary adjustments to images for more clarity.")

uploaded_file=st.sidebar.file_uploader(label="Upload Image",type=["jpg","jpeg","png"],key="1")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.subheader("Grayscale Image")
    st.image(image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),width=700)
