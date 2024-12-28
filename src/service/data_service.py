import os, time, base64, logging, random
import cv2 as cv
from datetime import datetime
from threading import Thread
from db.mongo_instance import MongoInstance

logger = logging.getLogger(__name__)

class DataService:
    def __init__(self, client:str = 'traffic_analysis') -> None:
        self.dmy = datetime.now().strftime("%d_%m_%y")
        self.client = MongoInstance(client)
        self.client.select_collection("vehicle")
    
    def _encode_image(self, frame):
        success, buffer = cv.imencode(".png", frame)

        if not success:
            logger.warning("Image encoding failed.")
            return None
        
        return base64.b64encode(buffer).decode('utf-8')
    
    def _assign_split(self):
        rand = random.random()
        if 0.0 <= rand < 0.7:
            return "train"
        elif 0.7 <= rand < 0.9:
            return "validation"
        else:
            return "test"
    
    def _prepare_data(self, item, frames, timestamp):
        base_image = self._encode_image(frames[0])
        anno_image = self._encode_image(frames[1])

        if not base_image or not anno_image:
            return None
        
        return {
            "confidence": item[0][0].item(),
            "class_id": item[1][0].item(),
            "class_name": item[2],
            "xyxy": item[3][0].tolist(),
            "xyxyn": item[4][0].tolist(),
            "xywh": item[5][0].tolist(),
            "xywhn": item[6][0].tolist(),
            "speed": item[7],
            "masks": item[8],
            "json": item[9],
            "dmy": self.dmy,
            "time_stamp": timestamp,
            "base_image": base_image,
            "annotated_image": anno_image,
            "split": self._assign_split(),
            "is_trained": False
        }

    def process_objects(self, frames, objects):
        for item in objects:
            logger.info("Starting image encoding")
            timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
            data = self._prepare_data(item, frames, timestamp)
            if data is None:
                continue

            logger.info("Transfering data")
            start_time = time.process_time()
            self.client.insert_data(data)
            stop_time = time.process_time()
            logger.info(f"Transfer complete, time {stop_time - start_time}")


    