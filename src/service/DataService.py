import os, csv, time, threading, shutil, pickle, base64, logging
from datetime import datetime
import cv2 as cv
from dotenv import load_dotenv, dotenv_values
from db.MongoInstance import MongoInstance
import numpy as np

load_dotenv()
logger = logging.getLogger(__name__)


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

class DataService():

    def __init__(self) -> None:
        self.dmy = datetime.now().strftime("%d_%m_%y")
        
        self.client = MongoInstance("traffic_analysis")
        self.client.select_collection("vehicle")
    
    @threaded
    def store_data(self, frames: list, objects: list = None):
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")

        if not self.client:
            self.client = MongoInstance("traffic_analysis")
            self.client.select_collection("vehicle")

        if objects:
            for object in objects:  

                logger.info("starting image encoding")
                success, buffer = cv.imencode(".png", frames[0])
                logger.info("Image encoding complete")

                if not success:
                    logger.info("Image encoding failed.")
                    continue

                image_b64 = base64.b64encode(buffer).decode("utf-8")
               
                logger.info("Transfering data")
                start_time = time.process_time()

                self.client.insert_data({
                    "confidence": object.conf.item(),
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
                    "image":image_b64
                                     })
                
                stop_time = time.process_time()
                logger.info(f"Transfer complete, time {stop_time-start_time}")
        