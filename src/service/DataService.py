import os
from datetime import datetime
import cv2 as cv
import csv
from dotenv import load_dotenv, dotenv_values
import zipfile
import shutil
import time
import subprocess

load_dotenv()

class DataService():

    def __init__(self) -> None:
        self.dmy = datetime.now().strftime("%d_%m_%y")

    def store_frame(self, frame: list, objects: list = None):
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
        filename = "unannotated_%s.png" % timestamp

        cv.imwrite("%s/unannotated_%s.png" % (os.getenv("OUT_DIR"), timestamp), frame[0])
        cv.imwrite("%s/annotated_%s.png" % (os.getenv("OUT_DIR"), timestamp), frame[1])

        return filename

    def store_video(self, frame: list, objects: list = None):
        filename = self.store_frame(frame, objects)
        self.store_metadata(objects, filename)
        # self.transfer_to_server()

    def store_metadata(self, objects: list = None, filename: str = None):
        fields = ["confidence", "class_id", "class_name", "data", "xmax", "ymax", "xmin", "ymin", "boxes", "time_stamp", "filename", "speed", "model_size"]
        path = '%s/traffic_%s.csv'% (os.getenv("OUT_DIR"), self.dmy)
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
        
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
                                     "time_stamp": timestamp,
                                     "filename": filename,
                                     "model_size": object.model_size})


    def zip_data(self, zip_path):
        """
        Zips all the .png files in the OUT_DIR directory into a zip file.

        Args:
            zip_path (str): The path to the zip file to be created.
        """
        with zipfile.ZipFile(zip_path, mode='w') as zfile:
            len_dir_path = len(os.getenv("OUT_DIR"))
            for root, _, files in os.walk(os.getenv("OUT_DIR")):
                for f in files:
                    if f.endswith(".png"):
                        zfile.write(os.path.join(root, f),  os.path.join(root, f)[len_dir_path:])
    
            
    def transfer_data(self):
        """
        Transfers data to the server.
        
        Transfers all .png files in the OUT_DIR directory to the server as a zip file if the size of the directory is greater than 1.5 GB.
        Transfers the csv file to the server at midnight.
        """
        if os.path.getsize(os.getenv("OUT_DIR")) >= 1.5 * 1024 * 1024 * 1024:
            self.zip_data("%s/traffic_data.zip"% "../data")
            subprocess.run(["scp", "/home/flor/Workspace/traffic_analysis/src/data/traffic_data.zip", os.getenv("SERVER_ADDRESS")])
        
        if datetime.datetime.now().hour == 0 and datetime.datetime.now().minute == 0:
            subprocess.run(["scp", "%s/traffic_%s.csv"% os.getenv("OUT_DIR"), self.dmy, os.getenv("SERVER_ADDRESS")])