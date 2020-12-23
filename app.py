import cv2
import math
import statistics
import numpy as np
import streamlit as st

from PIL import Image

def sharpen_img(image,
                kernel_size=(5, 5),
                sigma=1.0,
                amount=1.0,
                threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

st.title("Detect Testing app Circular-Shaped Objects")
st.markdown("This simple web application detects circular-shaped objects from uploaded images.")

st.sidebar.title("Image Adjustments")
st.sidebar.markdown("Perform any necessary adjustments to images for more clarity.")
sharpen_input = st.sidebar.number_input('Sharpen Image', 0, 10, 2)

uploaded_file=st.sidebar.file_uploader(label="Upload Image", type=["jpg","jpeg","png"], key="1")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = sharpen_img(image, amount=sharpen_input)
    st.subheader("Grayscale Image")
    st.image(image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),width=700)
