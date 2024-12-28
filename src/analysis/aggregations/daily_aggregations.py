from db.mongo_instance import MongoInstance
from service.data_storage import DataStorage

import logging

logger = logging.getLogger(__name__)

data_service = DataStorage()

client = MongoInstance("traffic_analysis")
data = client.select_collection("vehicle")

def analyze_daily(collection):
    """
    Perform comprehensive analysis of object classes in the database
    
    :param collection: PyMongo collection object
    :return: DataFrame with detailed class analysis
    """


    pipeline = [
        {
            "$addFields": {
                "day": {
                    "$toInt": { "$arrayElemAt": [{ "$split": ["$dmy", "_"] }, 0] }
                }
            }
        },
        {
            "$group": {
                "_id": {
                    "day": "$day",
                    "class_id": "$class_id",
                    "class_name": "$class_name"
                },
                "detections_count": { "$sum": 1 },

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
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "day": "$_id.day",
                "class_id": "$_id.class_id",
                "class_name": "$_id.class_name",
                "detections_count": 1,

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
        
    # Execute the pipeline
    results = collection.aggregate(pipeline)

    # Print results
    for doc in results:
        try:
            data_service.store_daily_data(doc)

        except Exception as e:
            print(f"Failed to store dayly data: {str(e)}")
            logger.error(f"Failed to store dayly data: {str(e)}")



def main_day():
    client = MongoInstance("traffic_analysis")
    client.select_collection("vehicle")
    
    # Perform analysis
    analysis_results = analyze_daily(client)
    
    # Display results
    print(analysis_results)


