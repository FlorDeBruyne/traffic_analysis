import numpy as np
import cv2 as cv
import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime
from collections import deque
import time, threading

from service.inference_service import Inference
from service.data_service import DataService

load_dotenv()
data = DataService()
inf = Inference()

def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper


class WebcamController():

    def __init__(self, device_id: int = 0, buffersize: int = 5):
        self.device_id = device_id
        self.timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
        self.buffersize = buffersize

        self.capture = self.camera_setup()
        self.recording = False    
        

    def camera_setup(self):
        """
        Setup up the webcam settings
        """
        assert cv.VideoCapture(self.device_id), "The input source is not accesible"
        capture = cv.VideoCapture(self.device_id)

        capture.set(cv.CAP_PROP_FRAME_WIDTH, int(os.getenv("FRAME_WIDTH")))
        capture.set(cv.CAP_PROP_FRAME_HEIGHT, int(os.getenv("FRAME_HEIGHT")))
   
        # This needs change because its season dependend, 
        # other way is to check on how much light there is and change the value depending on this
        if int(self.timestamp.split('_')[3]) >= 18 and int(self.timestamp.split('_')[3]) <= 5:
            capture.set(cv.CAP_PROP_AUTO_EXPOSURE, -10)
        else:
            capture.set(cv.CAP_PROP_AUTO_EXPOSURE, -10)

        return capture

    def stream_video(self):
        """
        Stream video from a webcam, process each frame to detect objects of interest,
        and display the results in real-time.
        """

        while True:
            ret, frame = self.capture.read()

            if not ret:
                print("Cannot retrieve a frame.")
                break

            # Process each yielded result from detect
            for detection, unannotated_frame, annotated_frame, objects in inf.detect(frame):
                if detection:
                    # Store the detected data
                    data.store_data([unannotated_frame, annotated_frame], objects)

                # Display the annotated frame
                cv.imshow("Frame", annotated_frame)

            if cv.waitKey(1) == ord('q'):
                break

        # Release resources when done
        self.capture.release()
        cv.destroyAllWindows()
