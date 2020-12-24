from cv2 import cv2
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


def get_circles(img, dp1, p1, p2, minr, maxr):

    rows = img.shape[0]

    circles = cv2.HoughCircles(
        image=img,
        method=cv2.HOUGH_GRADIENT,
        dp=dp1,
        minDist=(rows / 65),
        param1=p1,
        param2=p2,
        minRadius=minr,
        maxRadius=maxr)

    return circles

# Sidebar Headers
st.title("Detect Circular-Shaped Objects")
st.markdown("This simple web application detects circular-shaped objects from uploaded images.")

st.sidebar.title("Image Adjustments")
st.sidebar.markdown("If required, adjust images.")

file_types = ["JPG","JPEG","PNG"]

def main():

    # Image Uploader
    file = st.sidebar.file_uploader("Upload file", 
                                    type=file_types, 
                                    key="1"
                                 )

    show_file = st.empty()

    if not file:
        show_file.info('Please upload a file of type: '
                       + ', '.join(file_types))
        return


    # Filter Tools
    sharpen_input = st.sidebar.number_input('Sharpen Image', 0, 15, 1)


    # Model Tuning
    dp_list = list(np.arange(.025, 5, .01))
    dp_list = [ '%.2f' % elem for elem in dp_list ]

    dp1_input = st.sidebar.number_input(label='Accumulator Resolution', 
                                        min_value=.05, 
                                        max_value=5.0, 
                                        step=0.01, 
                                        value=1.20
                                    )

    p1_input = st.sidebar.number_input('Edge Detection', 
                                        min_value=1, 
                                        max_value=1000, 
                                        step=25, 
                                        value=400
                                    )

    p2_input = st.sidebar.number_input(label='Circle Detection Threshold', 
                                        min_value=0, 
                                        max_value=20, 
                                        step=1, 
                                        value=10
                                    )

    radius_input = st.sidebar.slider(label='Min - Max Radius', 
                                        min_value=0, 
                                        max_value=15, 
                                        value=(1, 8), 
                                        step=1
                                    )

    minRadius_input = int(radius_input[0])
    maxRadius_input = int(radius_input[1])


    #if st.sidebar.button("Show Options"):
    #    st.text("Button 1 pressed")
    #if st.sidebar.button("Hide Options"):
    #    st.text("Button 2 pressed")

    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, 1)


    import plotly.express as px
    import plotly.graph_objects as go

    fig = px.imshow(image)
    fig.update_layout(title='Uploaded Image', autosize=False, 
                      width=650,
                      height=500,
                      yaxis={'visible': False, 'showticklabels': False},
                      xaxis={'visible': False, 'showticklabels': False})

    st.plotly_chart(fig)

    # Image Adjustments 
    image = sharpen_img(image, amount=sharpen_input)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Circle Detecting Parameters
    circles = get_circles(gray,
                          dp1_input,
                          p1_input, 
                          p2_input, 
                          minRadius_input, 
                          maxRadius_input)

    # Circle Shape Calculations 
    circle_radius = circles[0, :, 2]  # Extract Radius Per Circle
    circle_count = len(circle_radius)  # Count Detected Circles

    #min_circle_size = round(min(circle_radius))  # Smallest Circle
    #max_circle_size = round(max(circle_radius))  # Biggest Circle

    avg_radius = (sum(circle_radius) / circle_count)  # Overall radius mean

    standard_deviation = statistics.stdev(circle_radius, avg_radius)

    min_circle_size = min(circle_radius)  # Smallest Circle
    max_circle_size = max(circle_radius)  # Biggest Circle

    stdv_places = 2 # Standard Deviations Away From Average

    stdv = (avg_radius - (stdv_places * standard_deviation))
    
    min_threshold = math.floor(stdv)
    max_threshold = math.ceil(stdv)

    # Get Specific-Sized Circle Count
    small_circles = len([i for i in circle_radius if i < min_threshold])
    big_circles = len([i for i in circle_radius if i > max_threshold])

    cimg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cimg = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)

    shape_size = 10

    for (x, y, r) in circles[0, :]:
        
        x = round(x)#.astype(int)
        y = round(y)#.astype(int)
        
        # Mark the center of the circle
        if r <= min_threshold:
            cv2.rectangle(cimg, (x-shape_size, y+shape_size), (x+shape_size, y-shape_size), (255, 0, 0), 1)
        
        # Mark circle outlines
        cv2.circle(cimg, (x, y), int(r), (0, 255, 0), 1)
        
        # Show overall metric summary 
        cv2.putText(cimg, 'Circles Count = ' + str(circle_count), (40, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
        cv2.putText(cimg, 'Potentially "Smaller" Circles = ' + str(small_circles), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)


    fig = px.imshow(cimg)
    fig.update_layout(title='Outcome Image', autosize=False, 
                      width=650,
                      height=500,
                      yaxis={'visible': False, 'showticklabels': False},
                      xaxis={'visible': False, 'showticklabels': False})
    st.plotly_chart(fig)
    

main()
