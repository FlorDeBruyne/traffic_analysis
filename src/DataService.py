import os
from datetime import datetime
import cv2 as cv
import csv
from dotenv import load_dotenv, dotenv_values

load_dotenv()

class DataService():

    def __init__(self) -> None:
        #NO NEED TO MAKE A VIDEOWRITER WHEN FUNCTIONANLITY IS ON HOLD

        # self.fourcc = cv.VideoWriter_fourcc(*'XVID')
        # self.out = cv.VideoWriter(filename = f'{os.getenv("OUT_DIR")}/output_{datetime.now().strftime("%d_%m_%y_%H_%M_%S")}.avi',
        #                           fourcc=self.fourcc,
        #                           fps=int(os.getenv("FPS")),
        #                           frameSize=(int(os.getenv("FRAME_WIDTH")), int(os.getenv("FRAME_HEIGHT"))))        
        pass

    def store_frame(self, frame):
        filename, timestamp = "unannotated_%s.png" % datetime.now().strftime("%d_%m_%y_%H_%M_%S"), datetime.now().strftime("%d_%m_%y_%H_%M_%S")
        cv.imwrite("%s/unannotated_%s.png" % (os.getenv("OUT_DIR"), datetime.now().strftime("%d_%m_%y_%H_%M_%S")), frame)
        return filename, timestamp

    def store_video(self, frame, objects: list = None):
        self.transfer_to_server()
        filename, timestamp = self.store_frame(frame)
        self.store_metadata(objects, filename, timestamp)
        #Pausing the video write till I have a better webcam 
        # self.out.write(frame)

    def store_metadata(self, objects: list = None, filename: str = None, timestamp: str = None):
        fields = ["confidence", "class_id", "class_name", "data", "xmax", "ymax", "xmin", "ymin", "time_stamp", "filename"]
        path = '%s/traffic_%s.csv'% (os.getenv("OUT_DIR"), datetime.now().strftime("%d_%m_%y"))
        
        if not os.path.exists(path):
            with open(path, 'w+') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fields)

        if objects:
            with open(path, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fields)

                for object in objects:                   
                    writer.writerow({"confidence": object.conf.item(),
                                     "class_id": object.cls_id.item(),
                                     "class_name": object.cls,
                                     "data": object.data,
                                     "xmax": object.coordinates[0].item(),
                                     "ymax": object.coordinates[1].item(),
                                     "ymin": object.coordinates[2].item(),
                                     "xmin": object.coordinates[3].item(),
                                     "time_stamp": timestamp,
                                     "filename": filename})


    def transfer_to_server(self):
        if len(os.listdir("%s/" % os.getenv("OUT_DIR"))) >= 50:
            #Transfer the files to a remote server to train a model on.
            pass