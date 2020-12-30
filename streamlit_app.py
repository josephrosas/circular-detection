import os
import math
import statistics
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from cv2 import cv2
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
import streamlit.report_thread as ReportThread

from PIL import Image

# ------------------------------------------------------- Application Headers

st.title('Detect Circular-Shaped Objects')
st.markdown('This simple web application detects circular-shaped objects from uploaded images.')

file_types = ['JPG', 'JPEG', 'PNG']

# ----------------------------------------------------------------- Functions

class SessionState(object):
    def __init__(self, **kwargs):
        """A new SessionState object.

        Parameters
        ----------
        **kwargs : any
            Default values for the session state.

        Example
        -------
        >>> session_state = SessionState(user_name='', favorite_color='black')
        >>> session_state.user_name = 'Mary'
        ''
        >>> session_state.favorite_color
        'black'
        """
        for key, val in kwargs.items():
            setattr(self, key, val)

@st.cache(allow_output_mutation=True)
def get_session(id, **kwargs):
    return SessionState(**kwargs)

def get(**kwargs):
    """Gets a SessionState object for the current session.

    Creates a new object if necessary.

    Parameters
    ----------
    **kwargs : any
        Default values you want to add to the session state, if we're creating a
        new one.

    Example
    -------
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    ''
    >>> session_state.user_name = 'Mary'
    >>> session_state.favorite_color
    'black'

    Since you set user_name above, next time your script runs this will be the
    result:
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    'Mary'

    """

    ctx = get_report_ctx()
    id = ctx.session_id
    return get_session(id, **kwargs)

# ---------------------------------------------------------- Sharpen Function

def imsharp(image, kernel_size=(5, 5), sigma=1.00, amount=1.00, threshold=0):
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

# ---------------------------------------------------------- Crop Input Image 

def imcrop(image, start=20, end=90): 
    """Crop spaces that do not reflect objects"""
    # Get height and width
    height, width = image.shape[:2]
    # Let's get the starting pixel coordiantes (top  left of cropping rectangle)
    start_row, start_col = int(height * .20), int(width * .20)
    # Let's get the ending pixel coordinates (bottom right)
    end_row, end_col = int(height * .90), int(width * .90)
    # Simply use indexing to crop out the rectangle we desire
    cropped = image[start_row:end_row, start_col:end_col]
    return cropped

# ----------------------------------------------------- Adjust Gamma Function 

def imgamma_adjust(image, gamma=1.0): 
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# ------------------------------------------------------ Apply Gamma Function 

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
    
# -------------------------------------------------- Gamma Threshold Function 

def imgamma_threshold(image, value=0.25): 
    """Get image threshold to a specific % of the maximum"""
    threshold = value * np.max(image)
    image[image <= threshold] = 0
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image

# ------------------------------------------------- Circle Detection Function 

def imcircles(img, dp1, p1, p2, min_radius, max_radius):
    """Circle detection using OpenCV's Hough Gradient"""
    rows = img.shape[0]
    circles = cv2.HoughCircles(
        image=img,
        method=cv2.HOUGH_GRADIENT,
        dp=dp1,
        minDist=rows / 65,
        param1=p1,
        param2=p2,
        minRadius=min_radius,
        maxRadius=max_radius)
    return circles

# --------------------------------------------------------- Uploader Function 

def st_file_selector(st_placeholder, path=".", label="Please, select a file/folder..."):
    # get base path (directory)
    base_path = "." if path is None or path is "" else path
    base_path = (
        base_path if os.path.isdir(base_path) else os.path.dirname(base_path)
    )
    base_path = "." if base_path is None or base_path is "" else base_path
    # list files in base path directory
    files = os.listdir(base_path)
    if base_path is not ".":
        files.insert(0, "..")
    files.insert(0, ".")
    selected_file = st_placeholder.selectbox(
        label=label, options=files, key=base_path
    )
    selected_path = os.path.normpath(os.path.join(base_path, selected_file))
    if selected_file is ".":
        return selected_path
    if os.path.isdir(selected_path):
        selected_path = st_file_selector(
            st_placeholder=st_placeholder, path=selected_path, label=label
        )
    return selected_path

# --------------------------------------------------------- Main App Function

def main():

# ----------------------------------------------------------- Uploader Widget 

    st.sidebar.title('Upload File Here')

    file = st.sidebar.file_uploader('', type=file_types)

    show_file = st.empty()

# ---------------------------------------------------------------------------

    if not file:
        show_file.info('See menu to upload a image')
        return

# ---------------------------------------------------------------------------

    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)

# ---------------------------------------------------------- Load Input Image

    original_image = cv2.imdecode(file_bytes, 1)

# ------------------------------------------------------- Display Input Image

    fig = px.imshow(original_image)

    display_options = {'visible': False, 'showticklabels': False} 

    fig.update_layout(
        title='Input Image',
        autosize=False,
        width=650,
        height=500,
        yaxis=display_options,
        xaxis=display_options,
        )
        
    st.plotly_chart(fig)

# ------------------------------------------------------- Menu Sidebar Headers 

    st.sidebar.title('Image Adjustments')
    st.sidebar.markdown('If required, adjust images.')

# ------------------------------------------------------- Image Default Values 

    gamma_default_value = 100
    sharpen_default_value = 5
    thresh_default_value = 0.25

# -------------------------------------------------------- Hough Circle Values 

    dp1_default_value = 1.20
    p1_default_value = 450
    p2_default_value = 8
    radius_default_values = (0, 8)

# ------------------------------------------------------------ Sharpen Widget 

    sharpen_input = st.sidebar.number_input('Sharpen', 
                         0, 15, sharpen_default_value)

# -------------------------------------------------------------- Gamma Widget 

    gamma_input = st.sidebar.number_input('Brightness', 0, 
                                100, gamma_default_value)

# ---------------------------------------------------------- Threshold Widget

    thresh_input = st.sidebar.number_input(label='Denoising',
                               min_value=.00, max_value=1.00, 
                               step=.01, value=thresh_default_value)

# ---------------------------------------------------------------------------

    dp1_input = st.sidebar.number_input(label='Accumulator Resolution',
            min_value=.05, max_value=5.0, step=.01, value=dp1_default_value)

# ---------------------------------------------------------------------------

    p1_input = st.sidebar.number_input('Edge Detection', min_value=1,
            max_value=1000, step=25, value=p1_default_value)

# ---------------------------------------------------------------------------

    p2_input = st.sidebar.number_input(label='Circle Detection Threshold',
                                min_value=0, max_value=20, step=1,
                                value=p2_default_value)

# ---------------------------------------------------------------------------

    radius_input = st.sidebar.slider(label='Min - Max Radius',
            min_value=0, max_value=15, value=radius_default_values, step=1)

    minRadius_input = int(radius_input[0])
    maxRadius_input = int(radius_input[1])

# ------------------------------------------------------ Circle Outline Widget

    visible_list = st.sidebar.radio('Show Circle Outlines', ['True', 'False'])
  
# --------------------------------------------------------------- Image Edits 

    image = original_image.copy()
    cropped = imcrop(image, start=20, end=80)
    gamma = imgamma_apply(cropped, amount=gamma_input)
    thresh = imgamma_threshold(gamma, value=thresh_input)
    sharpen = imsharp(thresh, amount=sharpen_input)
    gray = cv2.cvtColor(sharpen, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------------- Circle Detection 

    circles = imcircles(gray, dp1_default_value, 
                p1_default_value, p2_default_value, 
                minRadius_input, maxRadius_input)

# --------------------------------------------------------------- Circle Size 

    circle_radius = circles[0,:,2]
    circle_count = len(circle_radius)

# ------------------------------------------------------- Overall radius mean 

    avg_radius = sum(circle_radius) / circle_count  # 

# -------------------------------------------------------- Standard Deviation 

    standard_deviation = statistics.stdev(circle_radius, avg_radius)

# ----------------------------------------------------- Min / Max Circle Size 

    min_circle_size = min(circle_radius)
    max_circle_size = max(circle_radius)

# ------------------------------------- Standard Deviations Away From Average 

    stdv_places = 5

# -------------------------------------------- Standard Deviation Measurement

    stdv = avg_radius - (stdv_places * standard_deviation)

# ------------------------------------------------------- Min / Max Threshold 

    min_threshold = math.floor(stdv)
    max_threshold = math.ceil(stdv)

# ---------------------------------------------------- Min / Max Circle Count 

    small_circles = len([i for i in circle_radius if i < min_threshold])
    big_circles = len([i for i in circle_radius if i > max_threshold])

# --------------------------------------------- Small Circle's Label Box Size

    shape_size = 8

# ---------------------------------------------------------------------------
    # marked_img = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    # marked_img = cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB)
# ---------------------------------------------------------------------------

    marked_img = cropped.copy()

# ---------------------------------------------------------------------------

    for (x, y, r) in circles[0, :]:
        x = round(x)  # .astype(int)
        y = round(y)  # .astype(int)

# ---------------------------------------------------------------------------

        if r <= min_threshold:
            cv2.rectangle(marked_img, (x - shape_size, y + shape_size),
                          (x + shape_size, y - shape_size), (255, 0,
                          0), 2)

# ---------------------------------------------------------------------------

        if visible_list == 'True':
          # Mark circle outlines
            cv2.circle(marked_img, (x, y), int(r), (0, 255, 0), 2)

# -------------------------------------------------------------- Circle Count

        cv2.putText(
            marked_img,
            'Circles Count = ' + str(circle_count),
            (20, 285),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
            )

# ------------------------------------------------------- Small Circles Count 

        cv2.putText(
            marked_img,
            'Potential "Small" Circles = ' + str(small_circles),
            (20, 310),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
            )

# ------------------------------------------------------------- Display Image 

    fig = px.imshow(marked_img)

    fig.update_layout(
        title='Outcome Image',
        autosize=False,
        width=650,
        height=500,
        yaxis={'visible': False, 'showticklabels': False},
        xaxis={'visible': False, 'showticklabels': False})

    st.plotly_chart(fig)

# ----------------------------------------------------------------------- End

main()
