import cv2
import math
import statistics
import numpy as np
import streamlit as st

from PIL import Image
from skimage import io
from numpy import arange

import plotly.express as px
import plotly.graph_objects as go

def sharpen_img(
    image,
    kernel_size=(5, 5),
    sigma=1.0,
    amount=1.0,
    threshold=0,
    ):
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


def get_circles(
    img,
    dp1,
    p1,
    p2,
    minR,
    maxR,
    ):

    rows = img.shape[0]

    circles = cv2.HoughCircles(
        image=img,
        method=cv2.HOUGH_GRADIENT,
        dp=dp1,
        minDist=rows / 65,
        param1=p1,
        param2=p2,
        minRadius=minR,
        maxRadius=maxR,
        )

    return circles


def plotly_image(img):

    im_pil = Image.fromarray(img)

    # Create figure

    fig = go.Figure()

    # Constants

    img_width = 2100
    img_height = 1600
    scale_factor = 0.5

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.

    fig.add_trace(go.Scatter(x=[0, img_width * scale_factor], y=[0,
                  img_height * scale_factor], mode='markers',
                  marker_opacity=0))

    # Configure axes

    fig.update_xaxes(visible=False, range=[0, img_width * scale_factor])

    fig.update_yaxes(visible=False, range=[0, img_height
                     * scale_factor], scaleanchor='x')  # the scaleanchor attribute ensures that the aspect ratio stays constant

    # Add image

    fig.add_layout_image(dict(
        x=0,
        sizex=img_width * scale_factor,
        y=img_height * scale_factor,
        sizey=img_height * scale_factor,
        xref='x',
        yref='y',
        opacity=1.0,
        layer='below',
        sizing='stretch',
        source=im_pil,
        ))

    # Configure other layout

    fig.update_layout(width=img_width * scale_factor, height=img_height
                      * scale_factor, margin={
        'l': 0,
        'r': 0,
        't': 0,
        'b': 0,
        }, newshape=dict(line_color='blue'),
            title_text='Drag to add annotations - use modebar to change drawing tool'
            )

    # Disable the autosize on double click because it adds unwanted margins around the image

    fig.show(config={'doubleClick': 'reset', 'modeBarButtonsToAdd': [
        'drawline',
        'drawopenpath',
        'drawclosedpath',
        'drawcircle',
        'drawrect',
        'eraseshape',
        ]})
    
    
file_types = ['png', 'jpg']

def main():

    st.info(__doc__)
    st.markdown(STYLE, unsafe_allow_html=True)

    st.sidebar.subheader('Adjust Image')
    sharpen_input = st.sidebar.number_input('Sharpen Image', 0, 10, 2)
    
    dp_list = list(np.arange(.025, 5, .01))
    dp_list = ['%.2f' % elem for elem in dp_list]
    
    dp1_input = st.sidebar.number_input(label='dp', min_value=.05,
                                    max_value=5.0, step=0.01,
                                    value=1.23)
    param1_input = st.sidebar.number_input('param1', min_value=1,
            max_value=1000, step=25, value=200)
    param2_input = st.sidebar.number_input(label='param2', min_value=0,
            max_value=20, step=1, value=5)
    radius_input = st.sidebar.slider(label='Min/Max Radius', min_value=0,
                                     max_value=15, value=(2, 7), step=1)
    
    minRadius_input = int(radius_input[0])
    maxRadius_input = int(radius_input[1])
    
    all_filaments = st.sidebar.radio('Show All Circles Found', ['Yes', 'No'])
    
    file = st.file_uploader('Upload file', type=file_types)
    show_file = st.empty()

    if not file:
        show_file.info('Please upload a file of type: '
                       + ', '.join(file_types))
        return

    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, 1)
    image = sharpen_img(image, amount=sharpen_input)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    circles = get_circles(
        img,
        dp1_input,
        param1_input,
        param2_input,
        minRadius_input,
        maxRadius_input,
        )

    circle_radius = circles[0, :, 2]
    circle_count = len(circle_radius)

    min_circle_size = round(min(circle_radius))
    max_circle_size = round(max(circle_radius))

    avg_radius = round(sum(circle_radius) / circle_count)

    standard_deviation = statistics.stdev(circle_radius, avg_radius)

    min_threshold = math.floor(avg_radius - 2 * standard_deviation)
    max_threshold = math.ceil(avg_radius - 2 * standard_deviation)

    small_filaments = len([i for i in circle_radius if i < min_threshold])

    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    shape_size = 10

    for (x, y, r) in circles[0, :]:

        x = round(x).astype(int)
        y = round(y).astype(int)

        # draw the center of the circle
        if r <= min_threshold:
            cv2.rectangle(cimg, (x - shape_size, y + shape_size), (x
                          + shape_size, y - shape_size), (255, 0, 0), 1)

        # draw the outer circle
        cv2.circle(cimg, (x, y), int(r), (0, 255, 0), 1)

        # Print Small Filament Count Text
        cv2.putText(
            cimg,
            'Smaller Circles = ' + str(small_filaments),
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            .5,
            (255, 0, 0),
            2,
            )

        # Print Filament Count Text

        cv2.putText(
            cimg,
            'Total Circles = ' + str(circle_count),
            (40, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            .5,
            (255, 0, 0),
            2,
            )

    st.subheader('Output Image')
    st.image(cimg, width=690)

    # fig = px.imshow(cimg)
    # fig.show()

    # plotly_image(cimg)
    
main()
