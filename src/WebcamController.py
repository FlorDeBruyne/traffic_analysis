import numpy as np
import cv2 as cv
import os
from dotenv import load_dotenv
from datetime import datetime
from collections import deque
import time

from Inference import Inference
from DataService import DataService


load_dotenv()
data = DataService()
inf = Inference()


class WebcamController():

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.capture = self.camera_setup()
        self.buffer = deque(maxlen=30)
        self.recording = False
        self.record_start_time = None

    def camera_setup(self):
        assert cv.VideoCapture(self.device_id), "The input source is not accesible"
        capture = cv.VideoCapture(self.device_id)

        capture.set(cv.CAP_PROP_FRAME_WIDTH, int(os.getenv("FRAME_WIDTH")))
        capture.set(cv.CAP_PROP_FRAME_HEIGHT, int(os.getenv("FRAME_HEIGHT")))
        capture.set(cv.CAP_PROP_FPS, int(os.getenv("FPS")))
        capture.set(cv.CAP_PROP_EXPOSURE, +1)

        return capture


    def stream_video(self):
        """
        Stream video from a webcam, saves and records clips when an object of intrest is detected.
        """
        
        while True:
            ret, frame = self.capture.read()

            if not ret:
                print("Can not retrieve a frame.")
                break
            
            detection, model_frames, objects = inf.detect(frame)
            
            if detection == True:
                data.store_frame(frame)
                self.buffer.append(frame) # Store frame in buffer

                if not self.recording:
                    self.recording = True
                    self.record_start_time = time.time()

                if self.recording and (time.time() - self.record_start_time <= 2):
                    data.store_video(frame)
                else:
                    self.recording = False
                    self.buffer.clear()
                
            if self.recording:
                # Save buffered frames
                while self.buffer:
                    data.store_video(self.buffer.popleft())

            cv.imshow("Frame", frame)

            if cv.waitKey(1) == ord('q'):
                break
        
        self.capture.release()
        cv.destroyAllWindows()
        
    