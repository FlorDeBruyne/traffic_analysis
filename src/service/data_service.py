import os, time, base64, logging, threading
from datetime import datetime
import cv2 as cv
from PIL import Image
from dotenv import load_dotenv, dotenv_values
from db.mongo_instance import MongoInstance
from io import BytesIO
import numpy as np
import random
import albumentations as A


load_dotenv()
logger = logging.getLogger(__name__)


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

class DataService():

    def __init__(self, client:str, augment = False) -> None:
        self.dmy = datetime.now().strftime("%d_%m_%y")
        
        self.client = MongoInstance("traffic_analysis")
        self.client.select_collection(client)

        if augment == True:
            self._transformations()


    def _transformations(self):
        self.train_transform = A.Compose([
                    A.SmallestMaxSize(max_size=640),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                    A.RandomCrop(height=128, width=128),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ])
        
        self.val_transform = A.Compose([
            A.SmallestMaxSize(max_size=640),
            A.CenterCrop(height=128, widt=128),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])


    def _assign_split(self):
        """
        Assign a split to an object
        """
        rand = random.random()
        if 0.0 <= rand < 0.7:
            return "train"
        elif 0.7 <= rand < 0.9:
            return "validation"
        else:
            return "test"
    

    @threaded
    def store_inference_data(self, frames: list, objects: list = None):
        """
        Store send images to the mongodb collection.
        
        Args:
            frames (list): Images to be saved
            objects (list): Objects in these images
        """
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
                    "box_width": item[5][0].tolist()[2],
                    "box_height": item[5][0].tolist()[3],
                    "xywhn": item[6][0].tolist(),
                    "speed": item[7],
                    "masks": item[8],
                    "json": item[9],
                    "dmy": self.dmy,
                    "time_stamp": timestamp,
                    "base_image":base_image,
                    "annotated_image": anno_image,
                    "split": self._assign_split(),
                    "is_trained": False
                                     })
                
                stop_time = time.process_time()
                logger.info(f"Transfer complete, time {stop_time-start_time}")


    @threaded
    def store_yearly_data(self, input_doc):
        """
        Store aggregated yearly data to the MongoDB collection.

        Args:
            input_doc (dict): Aggregated yearly metrics to be saved.
        """
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")

        # Ensure the MongoDB client and collection are initialized
        if not self.client:
            self.client = MongoInstance("traffic_analysis")
            self.client.select_collection("yearly_metrics")  # Select or create the yearly data collection

        logger.info("Storing yearly data to the database.")
        try:
            start_time = time.process_time()

            # Insert the aggregated yearly data
            self.client.insert_data({
                "year": input_doc.get("year"),
                "class_id": input_doc.get("class_id"),
                "class_name": input_doc.get("class_name"),
                "detections_count": input_doc.get("detections_count"),
                "confidence_metrics": input_doc.get("confidence_metrics"),
                "speed_metrics": input_doc.get("speed_metrics"),
                "box_metrics": input_doc.get("box_metrics"),
                "time_stamp": timestamp  # Timestamp of when the data was stored
            })

            stop_time = time.process_time()
            logger.info(f"Yearly data storage complete, time taken: {stop_time - start_time}")

        except Exception as e:
            print(f"Failed to store yearly data: {str(e)}")
            logger.error(f"Failed to store yearly data: {str(e)}")


    @threaded
    def store_monthly_data(self, input_doc):
        """
        Store aggregated monthly data to the MongoDB collection.

        Args:
            input_doc (dict): Aggregated monthly metrics to be saved.
        """
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")

        # Ensure the MongoDB client and collection are initialized
        if not self.client:
            self.client = MongoInstance("traffic_analysis")
            self.client.select_collection("monthly_metrics")  # Select or create the monthly data collection

        logger.info("Storing monthly data to the database.")
        try:
            start_time = time.process_time()

            # Insert the aggregated monthly data
            self.client.insert_data({
                "month": input_doc.get("month"),
                "class_id": input_doc.get("class_id"),
                "class_name": input_doc.get("class_name"),
                "detections_count": input_doc.get("detections_count"),
                "confidence_metrics": input_doc.get("confidence_metrics"),
                "speed_metrics": input_doc.get("speed_metrics"),
                "box_metrics": input_doc.get("box_metrics"),
                "time_stamp": timestamp  # Timestamp of when the data was stored
            })

            stop_time = time.process_time()
            logger.info(f"monthly data storage complete, time taken: {stop_time - start_time}")

        except Exception as e:
            print(f"Failed to store monthly data: {str(e)}")
            logger.error(f"Failed to store monthly data: {str(e)}")


    @threaded
    def store_dayly_data(self, input_doc):
        """
        Store aggregated dayly data to the MongoDB collection.

        Args:
            input_doc (dict): Aggregated dayly metrics to be saved.
        """
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")

        # Ensure the MongoDB client and collection are initialized
        if not self.client:
            self.client = MongoInstance("traffic_analysis")
            self.client.select_collection("dayly_metrics")  # Select or create the dayly data collection

        logger.info("Storing dayly data to the database.")
        try:
            start_time = time.process_time()

            # Insert the aggregated dayly data
            self.client.insert_data({
                "day": input_doc.get("day"),
                "class_id": input_doc.get("class_id"),
                "class_name": input_doc.get("class_name"),
                "detections_count": input_doc.get("detections_count"),
                "confidence_metrics": input_doc.get("confidence_metrics"),
                "speed_metrics": input_doc.get("speed_metrics"),
                "box_metrics": input_doc.get("box_metrics"),
                "time_stamp": timestamp  # Timestamp of when the data was stored
            })

            stop_time = time.process_time()
            logger.info(f"dayly data storage complete, time taken: {stop_time - start_time}")

        except Exception as e:
            print(f"Failed to store dayly data: {str(e)}")
            logger.error(f"Failed to store dayly data: {str(e)}")


    @threaded
    def store_category_data(self, input_doc: dict):
        """
        Store aggregated category data to the MongoDB collection.

        Args:
            input_doc (dict): Aggregated category metrics to be saved.
        """
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")

        # Ensure the MongoDB client and collection are initialized
        if not self.client:
            self.client = MongoInstance("traffic_analysis")
            self.client.select_collection("category_metrics")  # Select or create the category data collection

        logger.info("Storing category data to the database.")
        try:
            start_time = time.process_time()

            self.client.insert_data({
                "class_name": input_doc.get("class_name"),
                "total_count": input_doc.get("total_count"),
                "trained_count": input_doc.get("trained_count"),
                "untrained_count": input_doc.get("untrained_count"),
                "confidence_metrics": input_doc.get("confidence_metrics"),
                "speed_metrics": input_doc.get("speed_metrics"),
                "box_metrics": input_doc.get("box_metrics"),
                "split_distribution": input_doc.get("split_distribution", []),
                "time_stamp": timestamp  # Timestamp of when the data was stored
            })

            stop_time = time.process_time()
            logger.info(f"Category data storage complete, time taken: {stop_time - start_time}")

        except Exception as e:
            logger.error(f"Failed to store category data: {str(e)}")
            print(f"Failed to store category data: {str(e)}")


    def extract_conditional_images(self, condition, output_dir: str):
        """
        Extracts all images that comply with a certain condition from the MongoDB collection
        to a local directory.
        
        Args:
            condition 
            output_dir (str): Directory to save the extracted images.
        """

        if not self.client:
            self.client = MongoInstance("traffic_analysis")
            self.client.select_collection("vehicle")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info("Fetching confident data from MongoDB...")
        documents = self.client.retrieve_data(query={"condition": {"$gt": 0.95}})

        if not documents:
            logger.info("No documents found in the database.")
            return
        print(type(documents))
        self.save_images(documents, output_dir)


    def extract_all_images(self, output_dir: str):
        """
        Extracts all images from the MongoDB collection and save's them to a local directory.
        
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


    def save_images_localy(self, documents, output_dir:str):
        """
        Save all preprocessed images to a output_dir.

        Args:
            documents: MongoDB documents.
            output_dir (str): Directory to save the extracted images to.
        """

        for idx, doc in enumerate(documents):
            base_image = doc.get("base_image")
            anno_image = doc.get("annotated_image")
            split = doc.get("split")
            class_name = doc.get("class_name", "UNKOWN")
            timestamp = doc.get("time_stamp", "UNKOWN")

            if not base_image or not anno_image:
                print(f"Document {idx} does not have an image.")
                logger.warning(f"Document {idx} does not have an image.")
                continue

            if not split:
                print(f"Document {idx} does not have an assigned split.")
                logger.warning(f"Document {idx} does not have an assigned split.")
                continue
                
            try:
                base_image = self.preprocess_images(base_image)

                anno_image_data = self.preprocess_images(anno_image)

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


    def preprocess_images(self,
                          encoded_image,
                            split: str,
                            target_size: tuple = (640, 640)):
        """
        Preprocess images by resizing and optionally augmenting.
        
        Args:
            encoded_image: encoded image
            target_size (tuple): Desired image size for model input
        
        return:
            list of preprocessed images
        """
        image_data = base64.b64decode(encoded_image)
        base_image = Image.open(BytesIO(image_data))
        resized_image = base_image.resize(target_size)
        
        if split == "train":
            resized_image = self.train_transform(image=resized_image)['image']
        if split == "valid":
            resized_image = self.val_transform(image=resized_image)['image']

        return resized_image