import json
import datetime
from pymongo import MongoClient
from db.mongo_instance import MongoInstance

def generate_coco_json(db_name, collection_name, output_file):
    # Connect to MongoDB
    client = MongoInstance()
    db = client[db_name]
    collection = db[collection_name]

    # Initialize COCO fields
    coco_data = {
        "info": {
            "description": "Generated COCO Dataset",
            "year": datetime.datetime.now().year,
            "version": "1.0",
            "contributor": "Automated Script",
            "date_created": datetime.datetime.now().date
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_map = {}  # Maps category names to IDs
    annotation_id = 1  # Unique ID for annotations

    # Retrieve data from MongoDB
    data = collection.find()
    for record in data:
        # Process each image record
        image_id = record["image_id"]
        file_name = record["file_name"]
        height = record["height"]
        width = record["width"]
        annotations = record.get("annotations", [])

        # Add to images list
        coco_data["images"].append({
            "id": image_id,
            "file_name": file_name,
            "height": height,
            "width": width
        })

        # Process annotations
        for ann in annotations:
            category_name = ann["category"]
            bbox = ann["bbox"]
            area = ann.get("area", bbox[2] * bbox[3])  # Default area if not provided

            # Assign a unique category ID
            if category_name not in category_map:
                category_id = len(category_map) + 1
                category_map[category_name] = category_id
                coco_data["categories"].append({
                    "id": category_id,
                    "name": category_name,
                    "supercategory": "none"
                })
            else:
                category_id = category_map[category_name]

            # Add to annotations list
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })
            annotation_id += 1

    # Write to output JSON file
    with open(output_file, "w") as f:
        json.dump(coco_data, f, indent=4)
    print(f"COCO dataset JSON saved to {output_file}")

# Example usage
generate_coco_json(
    db_name="traffic_analysis",
    collection_name="vehicle",
    output_file="coco_dataset.json"
)
