import os
from datetime import datetime
import cv2 as cv
import csv
from dotenv import load_dotenv, dotenv_values
import zipfile

load_dotenv()

class DataService():

    def __init__(self) -> None:
        #NO NEED TO MAKE A VIDEOWRITER, WHEN FUNCTIONANLITY IS ON HOLD
        # self.fourcc = cv.VideoWriter_fourcc(*'XVID')
        # self.out = cv.VideoWriter(filename = f'{os.getenv("OUT_DIR")}/output_{self.timestamp}.avi',
        #                           fourcc=self.fourcc,
        #                           fps=int(os.getenv("FPS")),
        #                           frameSize=(int(os.getenv("FRAME_WIDTH")), int(os.getenv("FRAME_HEIGHT"))))   

        self.timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
        self.dmy = datetime.now().strftime("%d_%m_%y")

    def store_frame(self, frame):
        filename = "unannotated_%s.png" % self.timestamp
        cv.imwrite("%s/unannotated_%s.png" % (os.getenv("OUT_DIR"), self.timestamp), frame)
        return filename

    def store_video(self, frame, objects: list = None):
        self.transfer_to_server()
        filename = self.store_frame(frame)
        self.store_metadata(objects, filename)
        #Pausing the video write till I have a better webcam 
        # self.out.write(frame)

    def store_metadata(self, objects: list = None, filename: str = None):
        fields = ["confidence", "class_id", "class_name", "data", "xmax", "ymax", "xmin", "ymin", "boxes", "time_stamp", "filename", "speed"]
        path = '%s/traffic_%s.csv'% (os.getenv("OUT_DIR"), self.dmy)
        
        #create the initial csv file
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
                                     "speed": object.speed,
                                     "boxes": object.boxes,
                                     "time_stamp": self.timestamp,
                                     "filename": filename})


    def zip_data(self, zip_path):
        with zipfile.ZipFile(zip_path, mode='w') as zfile:
            len_dir_path = len(os.getenv("OUT_DIR"))
            for root, _, files in os.walk(os.getenv("OUT_DIR")):
                for f in files:
                    zfile.write(os.path.join(root, f),  os.path.join(root, f)[len_dir_path:])

    def transfer_to_server(self):
        """
        Two options:
        - Transfer a certain amount of files when the required amount is reached (because of storage limitations)
        - Every 24hs zip the folder containing the images and transfer    
        """
        self.zip_data("%s/traffic_data.zip"%os.getenv("OUT_DIR"))
        pass
        # self.zip_data("/home/flor/Workspace/traffic_analysis/src")
        if len(os.listdir("%s/" % os.getenv("OUT_DIR"))) >= 4000:
            #Transfer the files to a remote server to train a model on.
            pass