import cv2 as cv
import streamlit as st
from dotenv import load_dotenv

from service.Inference import Inference
from service.DataService import DataService
from service.WebcamController import WebcamController

load_dotenv()
data = DataService()
inf = Inference()

def initialize_camera():
    """Initialize camera with proper settings for network access"""
    camera = cv.VideoCapture(0)
    if camera.isOpened():
        # Set lower resolution for better network performance
        camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv.CAP_PROP_FRAME_HEIGHT, 640)
        # Set lower FPS for better network performance
        camera.set(cv.CAP_PROP_FPS, 15)
    return camera

def stream_video_with_streamlit(capture: cv.VideoCapture):
    # Create placeholders for frames
    st.title("Traffic analysis Stream")
    st.text("Displaying live stream.")

    frame_placeholder = st.empty()

    stop_button = st.button("Stop Stream")

    try:
        while capture.isOpened() and not stop_button:

            ret, frame = capture.read()

            if not ret:
                st.error("Cannot retrieve frame from webcam.")
                break

            detection, unannotated_frame, annotated_frame, objects = inf.detect(frame)

            if detection:
                data.store_data([unannotated_frame, annotated_frame], objects)
                display_frame = annotated_frame
            else:
                display_frame = unannotated_frame

            frame_placeholder.image(
                cv.cvtColor(display_frame, cv.COLOR_BGR2RGB),
                caption="Video Stream",
                use_container_width=True,
                channels="RGB"
            )
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        capture.release()

    finally:
        st.error("Finnaly happend")
        if capture is not None:
            capture.release()

if __name__ == "__main__":
    capture = initialize_camera()
    stream_video_with_streamlit(capture)


