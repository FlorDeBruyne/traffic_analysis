from db.mongo_instance import MongoInstance
from service.data_service import DataService

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

data_service = DataService(client="category_metrics", augment=False)

client = MongoInstance("traffic_analysis")
client.select_collection("vehicle")

# Aggregation pipeline
def analyze_object_classes(collection):
    """
    Perform comprehensive analysis of object classes in the database
    
    :param collection: PyMongo collection object
    :return: DataFrame with detailed class analysis
    """
    category_pipeline = [
        # Stage 1: Group by class name and compute metrics
        {
            "$group": {
                "_id": "$class_name",
                
                # Detection Counts
                "total_count": {"$sum": 1},
                "trained_count": {"$sum": {"$cond": [{"$eq": ["$is_trained", True]}, 1, 0]}},
                "untrained_count": {"$sum": {"$cond": [{"$eq": ["$is_trained", False]}, 1, 0]}},
                
                # Confidence metrics
                "confidence_avg": { "$avg": "$confidence" },
                "confidence_max": { "$max": "$confidence" },
                "confidence_min": { "$min": "$confidence" },
                "confidence_stddev": { "$stdDevSamp": "$confidence" },

                # Speed metrics (preprocess, inference, postprocess)
                "preprocess_avg": { "$avg": "$speed.preprocess" },
                "preprocess_max": { "$max": "$speed.preprocess" },
                "preprocess_min": { "$min": "$speed.preprocess" },
                "preprocess_stddev": { "$stdDevSamp": "$speed.preprocess" },

                "inference_avg": { "$avg": "$speed.inference" },
                "inference_max": { "$max": "$speed.inference" },
                "inference_min": { "$min": "$speed.inference" },
                "inference_stddev": { "$stdDevSamp": "$speed.inference" },

                "postprocess_avg": { "$avg": "$speed.postprocess" },
                "postprocess_max": { "$max": "$speed.postprocess" },
                "postprocess_min": { "$min": "$speed.postprocess" },
                "postprocess_stddev": { "$stdDevSamp": "$speed.postprocess" },
                
                # Bounding box metrics
                "width_avg": {
                    "$avg": {
                        "$subtract": [
                            { "$arrayElemAt": ["$xyxy", 2] },
                            { "$arrayElemAt": ["$xyxy", 0] }
                        ]
                    }
                },
                "width_max": {
                    "$max": {
                        "$subtract": [
                            { "$arrayElemAt": ["$xyxy", 2] },
                            { "$arrayElemAt": ["$xyxy", 0] }
                        ]
                    }
                },
                "width_min": {
                    "$min": {
                        "$subtract": [
                            { "$arrayElemAt": ["$xyxy", 2] },
                            { "$arrayElemAt": ["$xyxy", 0] }
                        ]
                    }
                },
                "width_stddev": {
                    "$stdDevSamp": {
                        "$subtract": [
                            { "$arrayElemAt": ["$xyxy", 2] },
                            { "$arrayElemAt": ["$xyxy", 0] }
                        ]
                    }
                },
                "height_avg": {
                    "$avg": {
                        "$subtract": [
                            { "$arrayElemAt": ["$xyxy", 3] },
                            { "$arrayElemAt": ["$xyxy", 1] }
                        ]
                    }
                },
                "height_max": {
                    "$max": {
                        "$subtract": [
                            { "$arrayElemAt": ["$xyxy", 3] },
                            { "$arrayElemAt": ["$xyxy", 1] }
                        ]
                    }
                },
                "height_min": {
                    "$min": {
                        "$subtract": [
                            { "$arrayElemAt": ["$xyxy", 3] },
                            { "$arrayElemAt": ["$xyxy", 1] }
                        ]
                    }
                },
                "height_stddev": {
                    "$stdDevSamp": {
                        "$subtract": [
                            { "$arrayElemAt": ["$xyxy", 3] },
                            { "$arrayElemAt": ["$xyxy", 1] }
                        ]
                    }
                },
                
                # Split Distribution
                "split_distribution": {"$push": "$split"}
            }
        },
        
        # Stage 2: Sort by total count (descending order)
        {"$sort": {"total_count": -1}},
        
        # Stage 3: Project for a readable and structured output
        {
            "$project": {
                "class_name": "$_id",
                "_id": 0,
                "total_count": 1,
                "trained_count": 1,
                "untrained_count": 1,
                "confidence_metrics": 1,
                "size_metrics": 1,
                "processing_times": 1,
                "split_distribution": 1,

                "confidence_metrics": {
                    "average": "$confidence_avg",
                    "max": "$confidence_max",
                    "min": "$confidence_min",
                    "std_dev": "$confidence_stddev",
                    "variance": { "$pow": ["$confidence_stddev", 2] }
                },
                
                "speed_metrics": {
                    "preprocess": {
                        "average": "$preprocess_avg",
                        "max": "$preprocess_max",
                        "min": "$preprocess_min",
                        "std_dev": "$preprocess_stddev",
                        "variance": { "$pow": ["$preprocess_stddev", 2] }
                    },
                    "inference": {
                        "average": "$inference_avg",
                        "max": "$inference_max",
                        "min": "$inference_min",
                        "std_dev": "$inference_stddev",
                        "variance": { "$pow": ["$inference_stddev", 2] }
                    },
                    "postprocess": {
                        "average": "$postprocess_avg",
                        "max": "$postprocess_max",
                        "min": "$postprocess_min",
                        "std_dev": "$postprocess_stddev",
                        "variance": { "$pow": ["$postprocess_stddev", 2] }
                    }
                },

                "box_metrics": {
                    "width": {
                        "average": "$width_avg",
                        "max": "$width_max",
                        "min": "$width_min",
                        "std_dev": "$width_stddev",
                        "variance": { "$pow": ["$width_stddev", 2] }
                    },
                    "height": {
                        "average": "$height_avg",
                        "max": "$height_max",
                        "min": "$height_min",
                        "std_dev": "$height_stddev",
                        "variance": { "$pow": ["$height_stddev", 2] }
                    }
                }
            }
        }
    ]
    
    # Execute the aggregation pipeline
    results = list(collection.aggregate(category_pipeline))
    
    for doc in results:
        try:
            data_service.store_category_data(doc)

        except Exception as e:
            print(f"Failed to store category data: {str(e)}")
            logger.error(f"Failed to store category data: {str(e)}")


def main_cate():
    client = MongoInstance("traffic_analysis")
    client.select_collection("vehicle")
    
    # Perform analysis
    analysis_results = analyze_object_classes(client)
    
    # Display results
    print(analysis_results)
