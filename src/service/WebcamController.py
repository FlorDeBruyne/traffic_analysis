import numpy as np
import cv2 as cv
import os
from dotenv import load_dotenv
from datetime import datetime
from collections import deque
import time

from service.Inference import Inference
from service.DataService import DataService


load_dotenv()
data = DataService()
inf = Inference()


class WebcamController():

    def __init__(self, device_id: int = 0, buffersize: int = 5):
        self.device_id = device_id
        self.timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
        self.buffersize = buffersize

        self.capture = self.camera_setup()
        self.buffer = deque(maxlen=self.buffersize)
        self.recording = False
        self.record_start_time = None
        

    def camera_setup(self):
        assert cv.VideoCapture(self.device_id), "The input source is not accesible"
        capture = cv.VideoCapture(self.device_id)

        capture.set(cv.CAP_PROP_BUFFERSIZE, self.buffersize + 5)
        capture.set(cv.CAP_PROP_FRAME_WIDTH, int(os.getenv("FRAME_WIDTH")))
        capture.set(cv.CAP_PROP_FRAME_HEIGHT, int(os.getenv("FRAME_HEIGHT")))
        capture.set(cv.CAP_PROP_FPS, int(os.getenv("FPS")))
   
        if int(self.timestamp.split('_')[3]) >= 18 and int(self.timestamp.split('_')[3]) <= 5:
            capture.set(cv.CAP_PROP_EXPOSURE, +8)
        else:
            capture.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)

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
            
            #Detection == True if object of interest
            #model_frames are frames with annotations of the model
            #Objects is a list of DetectedObject instances
            detection, unannotated_frame, annotated_frame, objects = inf.detect(frame)
            
            if detection == True:
                data.store_video([unannotated_frame, annotated_frame], objects)
            
            frame = annotated_frame

            # cv.imshow("Frame", frame)

            if cv.waitKey(1) == ord('q'):
                break
        
        self.capture.release()
        cv.destroyAllWindows()
        
    