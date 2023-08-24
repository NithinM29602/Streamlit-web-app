import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av
import streamlit as st
import mediapipe as mp

html_code = '''
<style>
.heading_one{
text-align : center;
}
</style>
'''

# Increases the font size of the Slider label
st.markdown(
        """<style>
    div[class*="row-widget stSelectbox"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 20px;
    }
    
    div[class*="stSlider"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 20px;
    }
    
        </style>
        """, unsafe_allow_html=True)


st.markdown(html_code, unsafe_allow_html=True)
st.markdown("<h1 class='heading_one'> Face Anonymizer</h1>", unsafe_allow_html=True)
for i in range(5):
    st.write("")
select_item = st.selectbox("**Select the Model you want**",('Rectangular Face Blur', 'Face Region Blur'),0)
st.write("")
blur_value = st.slider('**Blur Value**', 1, 100, 10,key='slider_string')
def video_callback(frame):
    img_bgr = frame.to_ndarray(format='bgr24')
    ih, iw, _ = img_bgr.shape
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Converting the frame from BGR to RGB

    if select_item == 'Rectangular Face Blur':
        mp_face_detection = mp.solutions.face_detection

        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                output = face_detection.process(image_rgb)  # Processing the Image and Identifying the co-ordinates or landmarks of the face

                if output.detections:
                    for detection in output.detections:
                        bounding_box = detection.location_data.relative_bounding_box
                        xmin, ymin, w, h = bounding_box.xmin, bounding_box.ymin, bounding_box.width, bounding_box.height
                        x = int(xmin * iw)
                        y = int(ymin * ih)
                        w = int(w * iw)
                        h = int(h * ih)

                        # Blurring the face
                        img_bgr[y: y + h, x: x + w] = cv2.blur(img_bgr[y: y + h, x: x + w], (blur_value, blur_value))
                        return av.VideoFrame.from_ndarray(img_bgr, format='bgr24')

    else:
        # Create a FaceMesh instance adn Video Capture instance
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()

        # source = cv2.VideoCapture(0)

        # Loop until the 'q' key is pressed
        # while cv2.waitKey(1) != ord('q'):
        #     has_frame, frame = source.read()
        #     if not has_frame:
        #         break

        image_copy = image_rgb.copy()

        # Process the frame with FaceMesh
        outputs = face_mesh.process(image_rgb)
        final_img = image_rgb

        landmarks_list = list()
        if outputs.multi_face_landmarks:
            for landmarks in outputs.multi_face_landmarks:
                for landmark in landmarks.landmark:
                    x1, y1 = landmark.x, landmark.y
                    x = int(x1 * iw)
                    y = int(y1 * ih)
                    landmarks_list.append([x, y])  # Store face landmarks
                    # cv2.circle(image_rgb, (x,y), 3, ((0,0,255))) #It is used to mark the regions on the face

            # Extracting the outer region of the points without considering the input regions
            face_region_points = cv2.convexHull(np.array(landmarks_list), returnPoints=True)

            # You can use this to plot the polygon lines on the image, it represents the outer region
            # cv2.polylines(image_bgr,[face_region_points], isClosed=True, color=(0,0,255), thickness=3, lineType=cv2.LINE_AA)

            # Create a mask of the face region
            mask = np.zeros((ih, iw), np.uint8)
            cv2.fillConvexPoly(mask, face_region_points, (255, 255, 255))

            # Blur the image copy and Extract the foreground of the face
            image_copy = cv2.blur(image_copy, (blur_value, blur_value))
            # image_copy = cv2.medianBlur(image_copy, 31) # Applying median filter with a 5x5 kernel
            mask_foreground = cv2.bitwise_and(image_copy, image_copy, mask=mask)

            # Extract background
            cv2.fillConvexPoly(image_rgb, face_region_points, 0)

            # Combine foreground and background
            final_img = cv2.add(image_rgb, mask_foreground)

            # Display the result
            # cv2.imshow('Face Anonymizer', cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
            return av.VideoFrame.from_ndarray(final_img, format='rgb24')

        # Release video source and close windows
        # source.release()
        # cv2.destroyAllWindows()

webrtc_streamer(key='example2', video_frame_callback=video_callback, media_stream_constraints={'video':True, 'audio':False})
