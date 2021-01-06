# ---------------------------------------------------------------------------------------- LIBRARIES
import os
import io
import cv2
import math
import imutils
import statistics
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import ipywidgets as widgets
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
import streamlit.report_thread as ReportThread

from PIL import Image
from io import BytesIO
from scipy import ndimage

# ---------------------------------------------------------------------------------------- HEADERS

st.markdown("<h1 style='text-align: center; color:blue'>DEVELOPMENT ONLY</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Detect Circular-Shaped Objects</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Simple web app to detect circular-shaped objects from images.</h4>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>DEVELOPMENT ONLY</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------- UPLOADER 

def file_selector(st_placeholder, path=".", label="Please, select a file/folder..."):

    # get base path (directory)
    base_path = "." if path == None or path == "" else path
    base_path = (base_path if os.path.isdir(base_path) else os.path.dirname(base_path))
    base_path = "." if base_path == None or base_path == "" else base_path

    # list files in base path directory
    files = os.listdir(base_path)

    if base_path != ".":
        files.insert(0, "..")

    files.insert(0, ".")

    selected_file = st_placeholder.selectbox(label=label, 
                                             options=files, 
                                             key=base_path)

    selected_path = os.path.normpath(os.path.join(base_path, selected_file))

    if selected_file == ".":
        return selected_path

    if os.path.isdir(selected_path):
        selected_path = st_file_selector(st_placeholder=st_placeholder, 
                                        path=selected_path, label=label)
    return selected_path

def imsharp(image, kernel_size=(5, 5), sigma=1.00, amount=1.00, threshold=0,):
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

def imgamma_adjust(image, gamma=1.0): 
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def imgamma_apply(image, amount=1.5): 
            # loop over various values of gamma
    for gamma in np.arange(0.0, 3.5, 0.5):
        # ignore when gamma is 1 (there will be no change to the image)
        if gamma == 1:
            continue
        # apply gamma correction and show the images
        gamma = gamma if gamma > 0 else 0.1
        adjusted = imgamma_adjust(image, gamma=gamma)
        #cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return adjusted

def imgamma_threshold(image, value=0.25): 
    """Get image threshold to a specific % of the maximum"""
    threshold = value * np.max(image)
    image[image <= threshold] = 0
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image

def find_circles(image, dp=1.2, param1=300, param2=15, minRadius=0, maxRadius=9, dpi=42):
    """ finds the center of circular objects in image using hough circle transform

    Keyword arguments
    image -- uint8: numpy ndarray of a single image (no default).
    dp -- Inverse ratio of the accumulator resolution to the image resolution (default 1.7).
    minDist -- Minimum distance in pixel distance between the centers of the detected circles (default 100).
    param1 -- First method-specific parameter (default = 50).
    param2 -- Second method-specific parameter (default = 50).
    minRadius -- Minimum circle radius in pixel distance (default = 0).
    maxRadius -- Maximum circle radius in pixel distance (default = 0).

    Output
    center -- tuple: (x,y).
    radius -- int : radius.
    ERROR if circle is not detected. returns(-1) in this case    
    """
    
    rows = image.shape[0]
    
    circles = cv2.HoughCircles(image,
                               cv2.HOUGH_GRADIENT,
                               dp=dp,
                               minDist=(rows/dpi),
                               param1=param1,
                               param2=param2,
                               minRadius=minRadius,
                               maxRadius=maxRadius)
    if circles is not None:
        return(circles)
    else:
        raise ValueError(
            "ERROR! circle not detected try tweaking the parameters or the min and max radius")

# ---------------------------------------------------------------------------------------- MAIN APP 

def main():
    """ Main Application """

    file = st.file_uploader('', type=['JPG', 'JPEG', 'PNG'])

    show_file = st.empty()

# ---------------------------------------------------------------------------------------- UPLOADER 

    if not file:
        show_file.info('Upload Image To Detect Circles')
        return

    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)

    original_image = cv2.imdecode(file_bytes, 1)

# ---------------------------------------------------------------------------------------- IMAGE ADJUSTMENTS

    # Create mask image by copying original

    cv2.imwrite("image.png", original_image)
    image_png = cv2.imread("image.png")
    
    output = image_png.copy()
    image = image_png.copy()

    im_resize = cv2.resize(image, (500, 500))
    is_success, im_buf_arr = cv2.imencode(".png", im_resize)
    byte_im = im_buf_arr.tobytes()

    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    sharpen = imsharp(opening, amount=3)
    gamma = imgamma_apply(sharpen, amount=100)
    thresh = imgamma_threshold(gamma, value=0.50)
    
    # Convert Image To Grayscale
    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    # Find Image Row Count
    rows = gray.shape[0]

    # Define DPI (Dots Per Inch) Value
    dpi=42

    # Get Detected Circles
    circles = find_circles(
        
                           gray, 
                           dp=1.3, 
                           param1=450, 
                           param2=7, 
                           minRadius=1, 
                           maxRadius=9, 
                           dpi=42       
                    )

# ---------------------------------------------------------------------------------------- CIRCLE CALCULATION

    circles_radius = circles.copy()

    rcircles = np.uint16(np.around(circles_radius))

    if circles_radius is not None:
        circles_radius = np.round(circles_radius[0, :]).astype("int")

        # Count circles
        count=1
        r_mm_list = []
        
        for (x, y, r) in circles_radius:
            #Calculate radius in mm:
            r_mm = round((r/dpi), 5)
            r_mm_list.append(r_mm)

    r_mm_mean = statistics.mean(r_mm_list)
    r_mm_stdev = statistics.stdev(r_mm_list)
    stdev_place = (r_mm_mean - (3 * r_mm_stdev))

# ---------------------------------------------------------------------------------------- MARK CIRCLES

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print ("Number of circles:", len(circles))

        # Count circles
        count=1
        count_list = []
        r_mm_list = []
        radii_list = []
        small_circle_list = []
        
        for (x, y, r) in circles:
            #Calculate radius in mm:
            r_mm = round((r/dpi), 5)
            
            if r_mm <= stdev_place:
                cv2.circle(output, (x,y), r, (0, 255, 0), 1)
                # Add radius to center
                cv2.rectangle(output, (x-1, y-1), (x+1, y+1), (0,255,0), -1)
                small_circle_list.append(count)
            else:
                # Create outer circle
                cv2.circle(output, (x,y), r, (255, 0, 0), 1)
                # Create center rectangle
                cv2.rectangle(output, (x-1, y-1), (x+1, y+1), (255,0,0), -1)
            
            count_list.append(count)
            r_mm_list.append(r_mm)
            radii_list.append(r)

            count += 1

# ---------------------------------------------------------------------------------------- SEPARATOR 

    st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------- HEADER COUNT 

    count_circles = len(circles)
    count_small_cirlces = len(small_circle_list)

    st.subheader("Circles Detected: " + str(count_circles))
    st.subheader("Small Circles Detected: " + str(count_small_cirlces))

# ---------------------------------------------------------------------------------------- SEPARATOR 

    st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------- OUTPUT IMAGE 

    fig = px.imshow(output)

    fig.update_layout(
        title='Output Image',
        autosize=False,
        width=650,
        height=500,
        yaxis={'visible': False, 'showticklabels': False},
        xaxis={'visible': False, 'showticklabels': False})

    st.plotly_chart(fig)

# ---------------------------------------------------------------------------------------- CREATE DATAFRAME

#    df = pd.DataFrame()
#
#    df['circle_count'] = count_list
#    df['radius_mm'] = r_mm_list
#    df['radius_pixels'] = radii_list

    # plot structure
#    fig = px.bar(df, 
#                x='radius_mm', 
#                y='circle_count', 
#                #text='radius_mm',
#                #hover_data=['radius_mm'], 
#                color='radius_pixels'
#            )

#    fig.update_layout(
#        title_text='Input Image', # title of plot
#        yaxis={'visible': False, 'showticklabels': False}, # xaxis label
#        xaxis={'visible': False, 'showticklabels': False} # yaxis label    
#    )

#    st.plotly_chart(fig)

# ---------------------------------------------------------------------------------------- END OF APP

main()
