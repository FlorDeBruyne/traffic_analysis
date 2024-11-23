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
            for item in objects:  

                logger.info("starting image encoding")
                success, buffer = cv.imencode(".png", frames[0])
                logger.info("Image encoding complete")

                if not success:
                    logger.info("Image encoding failed.")
                    continue

                image_b64 = base64.b64encode(buffer).decode("utf-8")
               
                logger.info("Transfering data")
                start_time = time.process_time()

                # Needs more data: metadata (model_id, model size, type, inference type with speed),
                # segmentation (map50,75,95), isTrained = False when added to db
                # Make a split attribute -> [train, test, val] automatically make a entry either of these

                # 20% of images should be Test, 10% of images Validation and 70% of images will be Train split
                # This can be done through the len of the output

                # Make a other db for the evaluation entries of models, connected through model_id

                self.client.insert_data({
                            #         output.append([frame_results.boxes.xywh[0],
                            #    frame_results.boxes.xywhn[0],
                            #    frame_results.boxes.xyxy[0],
                            #    frame_results.boxes.xyxyn[0],
                            #     frame_results.keypoints,
                            #     frame_results.masks,
                            #     frame_results.obb, #LIST???
                            #     frame_results.orig_img,
                            #     frame_results.probs,
                            #     frame_results.speed[0], # preprocess
                            #     frame_results.speed[1], # inference
                            #     frame_results.speed[2], # postprocess
                            #     self.model.model])



                    # Boxes:
                    "xywh" : item[0],
                    "xywhn": item[1],
                    "xyxy": item[2],
                    "xyxyn": item[3],
                    "keypoints": item[4],
                    "masks": item[5],
                    "obb": item[6],
                    "orig"



                    "confidence": item.conf.item(),
                    "class_id": item.cls_id.item(),
                    "class_name": item.cls,
                    "data": pickle.dumps(item.data),
                    "xmax": item.coordinates[0].item(),
                    "ymax": item.coordinates[1].item(),
                    "ymin": item.coordinates[2].item(),
                    "xmin": item.coordinates[3].item(),
                    "speed": item.speed,
                    "boxes": item.boxes.tolist(),
                    "dmy": self.dmy,
                    "time_stamp": timestamp,
                    "image":image_b64,
                    'keypoints':item.keypoints,
                    "masks": item.masks
                                     })
                
                stop_time = time.process_time()
                logger.info(f"Transfer complete, time {stop_time-start_time}")
        