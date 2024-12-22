import os
import base64
import logging
import numpy as np
from PIL import Image
from io import BytesIO
from db.mongo_instance import MongoInstance
from service.data_augmentation import get_transformations
from pipelines.coco_pipeline import generate_coco_json
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, db_name="traffic_analysis", collection_name="vehicle", augment: bool = True):
        self.client = MongoInstance(db_name)
        self.client.select_collection(collection_name)

        if augment:
            self.train_transform, self.val_transform = get_transformations()

    def extract_untrained_images(self, output_dir: str, coco_output_file: str):
        """
        Extracts all images that have not been trained from the MongoDB collection,
        saves them to a local directory, and generates a COCO JSON file.
        
        Args:
            output_dir (str): Directory to save the extracted images.
            coco_output_file (str): Path to save the generated COCO JSON file.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info("Fetching untrained data from MongoDB...")
        documents = self.client.retrieve_data(query={"is_trained": False})

        if not documents:
            logger.info("No documents found in the database.")
            return

        self.save_images(documents, output_dir)
        self.prepare_coco_json(documents, output_dir, coco_output_file)

    def extract_all_images(self, output_dir: str, coco_output_file: str):
        """
        Extracts all images from the MongoDB collection, saves them to a local directory,
        and generates a COCO JSON file.
        
        Args:
            output_dir (str): Directory to save the extracted images.
            coco_output_file (str): Path to save the generated COCO JSON file.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info("Fetching data from MongoDB...")
        documents = self.client.fetch_all_documents()

        if not documents:
            logger.info("No documents found in the database.")
            return
        
        self.save_images(documents, output_dir)
        self.prepare_coco_json(documents, output_dir, coco_output_file)

    def save_images(self, documents, output_dir: str):
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
            class_name = doc.get("class_name", "UNKNOWN")
            timestamp = doc.get("time_stamp", "UNKNOWN")
            bbox = doc.get("xywh", [0, 0, 0, 0])

            if not base_image or not anno_image:
                logger.warning(f"Document {idx} does not have an image.")
                continue

            if not split:
                logger.warning(f"Document {idx} does not have an assigned split.")
                continue
                
            try:
                base_image = self.preprocess_images(base_image, split)

                class_dir = os.path.join(output_dir, class_name, split)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)

                base_image_path = os.path.join(class_dir, f"base_{timestamp}_id_{idx}.png")
                base_image.save(base_image_path, format="PNG")

                logger.info(f"Image {idx} saved to {base_image_path}")

                # Update document with file paths
                doc["file_name"] = base_image_path
                doc["annotations"] = [{"category": class_name, "bbox": bbox}]
                doc["height"] = base_image.height
                doc["width"] = base_image.width

            except Exception as e:
                logger.error(f"Failed to process document {idx}: {e}")

        logger.info("Image extraction complete.")

    def preprocess_images(self, encoded_image, split: str, target_size: tuple = (640, 640)):
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

        resized_image_np = np.array(resized_image)

        if split == "train":
            resized_image = self.train_transform(image=resized_image_np)['image']
        if split == "valid":
            resized_image = self.val_transform(image=resized_image_np)['image']

        resized_image = Image.fromarray(resized_image_np)

        return resized_image

    def prepare_coco_json(self, documents, output_dir: str, coco_output_file: str):
        """
        Prepare the COCO JSON file with the given documents.
        
        Args:
            documents: MongoDB documents.
            output_dir (str): Directory where images are saved.
            coco_output_file (str): Path to save the generated COCO JSON file.
        """
        for idx, doc in enumerate(documents):
            doc["image_id"] = idx + 1  # Assign a unique image ID

        generate_coco_json(documents, coco_output_file)
