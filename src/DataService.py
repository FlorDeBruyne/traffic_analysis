import os
from datetime import datetime
import cv2 as cv
from dotenv import load_dotenv, dotenv_values

load_dotenv()

class DataService():

    def __init__(self) -> None:
        self.fourcc = cv.VideoWriter_fourcc(*'XVID')
        self.out = cv.VideoWriter(filename = f'/home/flor/Train_images/output_{datetime.now().strftime("%d_%m_%y_%H_%M_%S")}.avi',
                                  fourcc=self.fourcc,
                                  fps=24.0,
                                  frameSize=(int(os.getenv("FRAME_WIDTH")), int(os.getenv("FRAME_HEIGHT"))))        

    def store_frame(self, frame):
        cv.imwrite("/home/flor/Train_images/frame_%s.png" % (datetime.now().strftime("%d_%m_%y_%H_%M_%S")), frame)

    def store_video(self, frame):
        self.init_writer()
        self.out.write(frame)

    def store_metadata(self, metadata):
        pass

    def transfer_to_server(self):
        if len(os.listdir("/home/flor/Train_images")) >= 30:
            print("transfer the files and a csv with metadata")