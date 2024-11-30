import os, csv, time, threading, base64, logging
from datetime import datetime
import cv2 as cv
from PIL import Image
from dotenv import load_dotenv, dotenv_values
from db.MongoInstance import MongoInstance
from io import BytesIO
import random

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

    # @threaded
    def assign_split(self):
        rand = random.random()
        if 0.0 <= rand < 0.7:
            return "train"
        elif 0.7 <= rand < 0.9:
            return "validation"
        else:
            return "test"
    
    # @threaded
    def store_data(self, frames: list, objects: list = None):
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")

        if not self.client:
            self.client = MongoInstance("traffic_analysis")
            self.client.select_collection("vehicle")

        if objects:
            for item in objects:  

                logger.info("starting image encoding")
                base_success, base_buffer = cv.imencode(".png", frames[0])
                anno_succes, anno_buffer = cv.imencode(".png", frames[1])

                if not base_success or not anno_succes:
                    logger.warning("Image encoding failed.")
                    continue

                logger.info("Image encoding complete")

                base_image = base64.b64encode(base_buffer).decode('utf-8')
                anno_image = base64.b64encode(anno_buffer).decode('utf-8')

                logger.info("Transfering data")
                start_time = time.process_time()

                self.client.insert_data({
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
                    "base_image":base_image,
                    "annotated_image": anno_image,
                    "split": self.assign_split(),
                    "is_trained": False
                                     })
                
                stop_time = time.process_time()
                logger.info(f"Transfer complete, time {stop_time-start_time}")

    

    def extract_conditional_images(self, confidence, output_dir: str):
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
        documents = self.client.retrieve_data(query={"confidence": {"$gt": 0.95}})

        if not documents:
            logger.info("No documents found in the database.")
            return
        print(type(documents))
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
            print("No documents found in the database.")
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
            base_image = doc.get("base_image")
            anno_image = doc.get("annotated_image")
            class_name = doc.get("class_name", "UNKOWN")
            timestamp = doc.get("time_stamp", "UNKOWN")

            if not base_image or not anno_image:
                print(f"Document {idx} does not have an image.")
                logger.warning(f"Document {idx} does not have an image.")
                continue

            try:
                base_image_data = base64.b64decode(base_image)
                base_image = Image.open(BytesIO(base_image_data))

                anno_image_data = base64.b64decode(anno_image)
                anno_image = Image.open(BytesIO(anno_image_data))

                class_dir = os.path.join(output_dir, class_name)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)

                base_image_path = os.path.join(class_dir, f"base_{timestamp}_id_{idx}.png")
                base_image.save(base_image_path, format="PNG")

                anno_image_path = os.path.join(class_dir, f"anno_{timestamp}_id_{idx}.png")
                anno_image.save(anno_image_path, format="PNG")

                print(f"Image {idx} saved to {base_image_path} and {anno_image_path}")

            except Exception as e:
                logger.error(f"Failed to process document {idx}: {e}")

        logger.info("Image extraction complete.")

        