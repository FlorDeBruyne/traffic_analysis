import os
import base64
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List, Iterator, Dict, Optional, Tuple
from dataclasses import dataclass
from db.mongo_instance import MongoInstance
from service.data_augmentation import get_transformations
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import datetime
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
        processed_docs = []
        for doc in documents:
            if doc is not None:
                if hasattr(doc.image_id, '__str__'):
                    doc.image_id = str(doc.image_id)
                processed_docs.append(doc)

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
            "id": str(doc.image_id),
            "file_name": doc.file_name,
            "height": doc.height,
            "width": doc.width
        }

    def _create_annotation_entry(self, doc: ImageDocument) -> Dict:
        annotation = {
            "id": self.next_annotation_id,
            "image_id": str(doc.image_id),
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
        self.coco_formatter = COCOFormatter()

        if augment:
            self.train_transform, self.val_transform = get_transformations()
        
        # self.target_size = (640, 640)vehicle
        self.batch_size = 10

    def process_dataset(self, output_dir: str, coco_output_file: str, untrained_only: bool = True):
        """Main processing pipeline"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # coco_data = self._initialize_coco_data()

        processed_documents = []

        query = {"is_trained": False} if untrained_only else {}
        total_documents = self.client.count_documents(query)
        processed_count = 0

        if not total_documents:
            logger.info("No Documents found in the database.")
            return
        
        try:
            for batch in self._get_document_batch(query):
                batch_docs = self._process_batch(batch, output_dir)
                processed_documents.extend([doc for doc in batch_docs if doc is not None])
                processed_count += len(batch)
                logger.info(f"Processed {processed_count}/{total_documents} images")
        
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise

        finally:
            if processed_documents:
                coco_data = self.coco_formatter.create_coco_format(processed_documents)
                self._save_coco_json(coco_data, coco_output_file)
        

    def _get_document_batch(self, query: Dict) -> Iterator[list]:
        """Get documents in batches to prevent memory overload"""
        cursor = self.client.find(query)
        current_batch = []

        try:
            for doc in cursor:
                current_batch.append(doc)
                if len(current_batch) >= self.batch_size:
                    yield current_batch
                    current_batch = []
            
            if current_batch:
                yield current_batch
        
        finally:
            cursor.close()


    def _process_batch(self, batch: list, output_dir: Path) -> List[ImageDocument]:
        """Process a batch of documents"""
        processed_docs = []

        for doc in batch:
            try:
                processed_doc = self._process_single_document(doc, output_dir)
                if processed_doc:
                    image_doc = ImageDocument(
                        image_id=processed_doc["image_id"],
                        base_image="",  # Empty as we don't need to store the image data
                        class_name=processed_doc["class_name"],
                        split=processed_doc["split"],
                        timestamp=processed_doc["timestamp"],
                        bbox=processed_doc["bbox"],
                        height=processed_doc["height"],
                        width=processed_doc["width"],
                        file_name=processed_doc["file_name"]
                    )
                    processed_docs.append(image_doc)

                
            except Exception as e:
                logger.error(f"Error processing document {doc.get('_id')}: {e}")
                continue
            
            finally:
                if 'base_image' in doc:
                    doc['base_image'] = None
                if 'annotated_image' in doc:
                    doc['annotated_image'] = None
        
        return processed_docs

    def _maintain_aspect_ratio(self, image: Image.Image, max_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """
        Resize image while maintaining aspect ratio.
        If max_size is provided, ensure image doesn't exceed these dimensions.
        """
        width, height = image.size
        
        if max_size:
            max_width, max_height = max_size
            # Calculate scaling factor to fit within max dimensions
            scale = min(max_width/width, max_height/height)
            
            if scale < 1:  # Only resize if image is too large
                new_width = int(width * scale)
                new_height = int(height * scale)
                return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image

    def _preprocess_image(self, image: Image.Image, split: str) -> Image.Image:
        """
        Preprocess a single image with proper type and size handling
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Keep original size or maintain aspect ratio if needed
        processed_image = self._maintain_aspect_ratio(image)
        
        # Convert to numpy for transformations
        image_array = np.array(processed_image)
        
        # Apply transformations if needed
        transformed_image = image_array
        try:
            if split == "train" and hasattr(self, 'train_transform'):
                transformed = self.train_transform(image=image_array)
                transformed_image = transformed['image']
            elif split == "validation" and hasattr(self, 'val_transform'):
                transformed = self.val_transform(image=image_array)
                transformed_image = transformed['image']
        except Exception as e:
            logger.warning(f"Transform failed, using original image: {e}")
            transformed_image = image_array

        # Ensure correct data type
        if transformed_image.dtype != np.uint8:
            transformed_image = (transformed_image * 255).clip(0, 255).astype(np.uint8)

        return Image.fromarray(transformed_image, mode='RGB')

    def _process_single_document(self, doc: Dict, output_dir: Path) -> Optional[Dict]:
        """Process a single document with proper size handling"""
        try:
            base_image = doc.get("base_image")
            if not base_image:
                return None

            doc_info = {
                "class_name": doc.get("class_name", "UNKNOWN"),
                "split": doc.get("split", "train"),
                "timestamp": doc.get("time_stamp", "UNKNOWN"),
                "bbox": doc.get("xywh", [0, 0, 0, 0]),
                "image_id": doc.get("_id")
            }

            # Decode and process image
            image_data = base64.b64decode(base_image)
            with Image.open(BytesIO(image_data)) as img:
                try:
                    processed_img = self._preprocess_image(img, doc_info["split"])

                except Exception as e:
                    logger.error(f"Image preprocessing failed: {e}")
                    processed_img = self._maintain_aspect_ratio(img)  # Fallback to simple resize

                # Save image with original dimensions
                class_dir = output_dir / doc_info["class_name"] / doc_info["split"]
                class_dir.mkdir(parents=True, exist_ok=True)
                
                file_name = f"base_{doc_info['timestamp']}_id_{doc_info['image_id']}.png"
                file_path = class_dir / file_name
                
                # Save with high quality
                processed_img.save(file_path, format="PNG", quality=100)
                
                doc_info.update({
                    "file_name": str(file_path),
                    "height": processed_img.height,
                    "width": processed_img.width
                })

            return doc_info

        except Exception as e:
            logger.error(f"Failed to process document: {e}")
            return None


    def _initialize_coco_data(self) -> Dict:
        """Initialize COCO data structure"""
        return {
            "info": {
                "description": "Generated COCO Dataset",
                "year": datetime.datetime.now().year,
                "version": "1.0",
                "contributor": "Automated Script",
                "date_created": datetime.datetime.now().date().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": {}
        }

    def _update_coco_data(self, coco_data: Dict, doc_info: Dict):
        """Update COCO data with processed document information"""
        coco_data["images"].append({
            "id": doc_info["image_id"],
            "file_name": doc_info["file_name"],
            "height": doc_info["height"],
            "width": doc_info["width"]
        })

        category_name = doc_info["class_name"]
        if category_name not in coco_data["categories"]:
            category_id = len(coco_data["categories"]) + 1
            coco_data["categories"][category_name] = {
                "id": category_id,
                "name": category_name,
                "supercategory": "none"
            }

        coco_data["annotations"].append({
            "id": len(coco_data["annotations"]) + 1,
            "image_id": doc_info["image_id"],
            "category_id": coco_data["categories"][category_name]["id"],
            "bbox": doc_info["bbox"],
            "area": doc_info["bbox"][2] * doc_info["bbox"][3],
            "iscrowd": 0
        })

    def _save_coco_json(self, coco_data: Dict, output_file: str):
        """
        Save COCO data to JSON file.

        Parameters:
        coco_data (Dict): The COCO formatted data to save.
        output_file (str): The path to the output JSON file.

        Returns:
        None
        """
        class COCOEncoder(json.JSONEncoder):
            def default(self, obj):
                from bson import ObjectId
                if isinstance(obj, ObjectId):
                    return str(obj)
                if isinstance(obj, Path):
                    return str(obj)
                return super().default(obj)

        # coco_data["categories"] = list(coco_data["categories"].values())
        
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=4, cls=COCOEncoder)
        logger.info(f"COCO dataset JSON saved to {output_file}")