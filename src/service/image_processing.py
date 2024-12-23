import os
import base64
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List, Dict, Optional
from dataclasses import dataclass
from db.mongo_instance import MongoInstance
from service.data_augmentation import get_transformations
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)

@dataclass
class ImageDocument:
    """Data class to hold image document information."""
    image_id: int
    base_image: str
    class_name: str
    split: str
    timestamp: str
    bbox: List[float]
    height: Optional[int] = None
    width: Optional[int] = None
    file_name: Optional[str] = None


class COCOFormatter:
    """Handles COCO format coversion"""
    def __init__(self):
        self.categories = {}
        self.next_category_id = 1
        self.next_annotation_id = 1

    def get_category_id(self, category_name: str):
        if category_name not in self.categories:
            self.categories[category_name] = {
                "id": self.next_category_id,
                "name": category_name,
                "supercategory": "none"
            }
            self.next_category_id += 1
        
        return self.categories[category_name]["id"]
    

    def create_coco_format(self, documents: List[ImageDocument]) -> Dict:
        """Convert documents to COCO format"""
        return {
            "info": self._create_info(),
            "licenses": [],
            "images": [self._create_image_entry(doc) for doc in documents],
            "annotations": [self._create_annotation_entry(doc) for doc in documents],
            "categories": list(self.categories.values())
        }


    def _create_info(self) -> Dict:
        from datetime import datetime
        return {
            "description": "Generated COCO Dataset",
            "year": datetime.now().year,
            "version": "1.0",
            "contributor": "Automated Script",
            "date_created": datetime.now().date().isoformat()
        }

    def _create_image_entry(self, doc: ImageDocument) -> Dict:
        return {
            "id": doc.image_id,
            "file_name": doc.file_name,
            "height": doc.height,
            "width": doc.width
        }

    def _create_annotation_entry(self, doc: ImageDocument) -> Dict:
        annotation = {
            "id": self.next_annotation_id,
            "image_id": doc.image_id,
            "category_id": self.get_category_id(doc.class_name),
            "bbox": doc.bbox,
            "area": doc.bbox[2] * doc.bbox[3], # Width * height
            "iscrowd": 0
        }
        self.next_annotation_id += 1
        return annotation
 

class ImageProcessor:
    def __init__(self, db_name="traffic_analysis", collection_name="vehicle", augment: bool = True):
        self.client = MongoInstance(db_name)
        self.client.select_collection(collection_name)

        if augment:
            self.train_transform, self.val_transform = get_transformations()
        
        self.target_size = (640, 640)


    def process_dataset(self, output_dir: str, coco_output_file: str, untrained_only: bool = True):
        """Main processing pipeline"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        query = {"is_trained": False} if untrained_only else {}
        documents = self.client.retrieve_data(query)

        if not documents:
            logger.info("No Documents found in the database.")
            return
        
        image_documents = self._prepare_documents(documents)

        self._process_images_parallel(image_documents, output_dir)

        self._generate_coco_json(image_documents, output_dir)

    def _prepare_documents(self, raw_documents: List[Dict]) -> List[ImageDocument]:
        """Convert raw MongoDB documents tot ImageDocument objects"""
        return [
            ImageDocument(
                image_id=idx + 1,
                base_image=doc["base_image"],
                class_name=doc.get("class_name", "UNKNOWN"),
                split=doc.get("split", "train"),
                timestamp=doc.get("time_stamp", "UNKNOWN"),
                bbox=doc.get("xywh", [0.0, 0.0, 0.0, 0.0])
            )
            for idx, doc in enumerate(raw_documents)
            if doc.get("base_image")
        ]


    def _process_single_image(self, doc: ImageDocument, output_dir: Path) -> Optional[ImageDocument]:
        """Process a single image document"""
        try:
            image_data = base64.b64decode(doc.base_image)
            image = Image.open(BytesIO(image_data))
            processed_image = self._preprocess_image(image, doc.split)

            class_dir = output_dir / doc.class_name / doc.split
            class_dir.mkdir(parents=True, exist_ok=True)
            
            file_name = f"base_{doc.timestamp}_id_{doc.image_id}.png"
            file_path = class_dir / file_name
            processed_image.save(file_path, format="PNG")

            doc.file_name = str(file_path)
            doc.height = processed_image.height
            doc.width = processed_image.width

            return doc

        except Exception as e:
            logger.error(f"Failed to process image {doc.image_id}: {e}")
            return None


    def _process_images_parallel(self, documents: List[ImageDocument], output_dir: Path):
        """Process images in parallel using ThreadPoolExecutor"""
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._process_single_image, doc, output_dir)
                for doc in documents
            ]

            processed_docs = []
            for future in tqdm(futures, desc="Processing images"):
                if doc := future.result():
                    processed_docs.append(doc)

        return processed_docs


    def _preprocess_image(self, image: Image.Image, split: str) -> Image.Image:
        """Preprocess a single image"""
        resized_image = image.resize(self.target_size)
        image_array = np.array(resized_image)

        if split == "train" and hasattr(self, 'train_transform'):
            image_array = self.train_transform(image=image_array)['image']
        elif split == "valid" and hasattr(self, 'val_transform'):
            image_array = self.val_transform(image=image_array)['image']

        return Image.fromarray(image_array)


    def _generate_coco_json(self, documents: List[ImageDocument], output_file: str):
        """Generate COCO JSON file"""
        formatter = COCOFormatter()
        coco_data = formatter.create_coco_format(documents)
        
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=4)
        logger.info(f"COCO dataset JSON saved to {output_file}")