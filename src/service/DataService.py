import os, csv, time, threading, shutil, pickle, base64, logging
from datetime import datetime
import cv2 as cv
import Image
from dotenv import load_dotenv, dotenv_values
from db.MongoInstance import MongoInstance

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

                self.client.insert_data({
                    "confidence": item[0][0].item(),
                    "class_id": item[1][0].item(),
                    "class_name": item[2][0],
                    "xyxy": item[3][0].tolist(),
                    "xyxyn": item[4][0].tolist(),
                    "xywh": item[5][0].tolist(),
                    "xywhn": item[6][0].tolist(),
                    "speed": item[7],
                    "masks": item[8],
                    "json": item[9],
                    "dmy": self.dmy,
                    "time_stamp": timestamp,
                    "image":image_b64,
                                     })
                
                stop_time = time.process_time()
                logger.info(f"Transfer complete, time {stop_time-start_time}")


    def extract_confident_images(self, confidence: float, output_dir: str):
        """
        Extracts all images that comply's with a certain confidence from the MongoDB collection.
        
        Args:
            output_dir (str): Directory to save the extracted images.
        """

        if not self.client:
            self.client = MongoInstance("traffic_analysis")
            self.client.select_collection("vehicle")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info("Fetching confident data from MongoDB...")
        documents = self.client.retrieve_data(query={"confidence": {"$gt": confidence}})

        self.save_images(documents, output_dir)


    def extract_all_images(self, output_dir: str):
        """
        Extracts all images from the MongoDB collection.
        
        Args:
            output_dir (str): Directory to save the extracted images.
        """

        if not self.client:
            self.client = MongoInstance("traffic_analysis")
            self.client.select_collection("vehicle")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info("Fetching data form MongoDB...")
        documents = self.client.fetch_all_documents()

        if not documents:
            logger.info("No documents found in the database.")
            return
        
        self.save_images(documents, output_dir)


    def save_images(self, documents, output_dir:str):
        """
        Save all images to a output_dir.

        Args:
            documents: MongoDB documents.
            output_dir (str): Directory to save the extracted images to.
        """

        for idx, doc in enumerate(documents):
            image_b64 = doc.get("image")
            class_name = doc.get("class_name", "")
            timestamp = doc.get("time_stamp", "")

            if not image_b64:
                logger.warning(f"Document {idx} does not have an image.")
                continue

            try:
                image_data = base64.b64decode(image_b64)
                image = Image.open(image_data)

                class_dir = os.path.join(output_dir, class_name)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)

                image_path = os.path.join(class_dir, f"{timestamp}_{idx}.png")
                image.save(image_path, format="PNG")

                logger.info(f"Image {idx} saved to {image_path}")

            except Exception as e:
                logger.error(f"Failed to process document {idx}: {e}")

        logger.info("Image extraction complete.")

        