import os, csv, shutil, time, threading, zipfile
from datetime import datetime
import cv2 as cv
from dotenv import load_dotenv, dotenv_values
from db.MongoInstance import MongoInstance
import numpy as np
import shutil
import pickle


load_dotenv()
client = MongoInstance("traffic_analysis")
client.select_collection("vehicle")

def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

class DataService():

    def __init__(self) -> None:
        self.dmy = datetime.now().strftime("%d_%m_%y")
                    
    def store_data(self, frames: list, objects: list = None):
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")

        if objects:
            for object in objects:  
               
                client.insert_data({"confidence": object.conf.item(),
                                     "class_id": object.cls_id.item(),
                                     "class_name": object.cls,
                                     "data": pickle.dumps(object.data),
                                     "xmax": object.coordinates[0].item(),
                                     "ymax": object.coordinates[1].item(),
                                     "ymin": object.coordinates[2].item(),
                                     "xmin": object.coordinates[3].item(),
                                     "speed": object.speed,
                                     "boxes": object.boxes.tolist(),
                                     "dmy": self.dmy,
                                     "time_stamp": timestamp,
                                     "image":frames[0]})
    
    # def _remove_original(self):
    #     for root, _, files in os.walk(self.OUT_DIR):
    #         for f in files:
    #             if f.endswith(".png"):
    #                 os.remove(os.path.join(self.OUT_DIR, f))

    
    # # @threaded
    # def transfer_data(self, zipped=False):
    #     """
    #     Transfers data to the server.
        
    #     Transfers all .png files in the OUT_DIR directory to the server as a zip file if the size of the directory is greater than 1.5 GB.
    #     Transfers the csv file to the server at midnight.
    #     """
    #     size = self._get_size()

    #     if size >= 0.5 * 1024 * 1024:
    #         if not os.path.exists(self.ZIP_PLACE):
    #             print("[TRANSFER] Starting zipping")
    #             zipped = self._zip_data(self.ZIP_PLACE)
    #             print("[TRANSFER] Completed zipping")
        
    #     if zipped:
    #         # self._remove_original()

    #         process = client.data_transfer(self.ZIP_PLACE)

    #         if not process:
    #             print("[TRANSFER] The transfer did not work")
    #             return
            
    #         print("[TRANSFER] Starting the transfer process")
        
    #     print("[TRANSFER] Not to big")

    #     if datetime.now().hour == 0 and datetime.now().minute == 0:
    #         client.data_transfer("%s/traffic_%s.csv"% self.OUT_DIR, self.dmy)
    #         print("[TRANSFER] CSV is being transfered")