from db.mongo_instance import MongoInstance
import numpy as np
import pandas as pd

client = MongoInstance("traffic_analysis")
client.select_collection("vehicle")

# Aggregation pipeline
def analyze_object_classes(collection):
    """
    Perform comprehensive analysis of object classes in the database
    
    :param collection: PyMongo collection object
    :return: DataFrame with detailed class analysis
    """
    pipeline = [
        # Stage 1: Group by class and calculate various metrics
        {
            "$group": {
                "_id": "$class_name",
                
                # Basic Counting Metrics
                "total_count": {"$sum": 1},
                
                # Confidence Metrics
                "avg_confidence": {"$avg": "$confidence"},
                "min_confidence": {"$min": "$confidence"},
                "max_confidence": {"$max": "$confidence"},
                "confidence_std_dev": {"$stdDevPop": "$confidence"},
                
                # Size Metrics (using xywhn normalized coordinates)
                "avg_width": {"$avg": {"$arrayElemAt": ["$xywhn", 2]}},
                "avg_height": {"$avg": {"$arrayElemAt": ["$xywhn", 3]}},
                "avg_area": {"$avg": {
                    "$multiply": [
                        {"$arrayElemAt": ["$xywhn", 2]},
                        {"$arrayElemAt": ["$xywhn", 3]}
                    ]
                }},
                
                # Bounding Box Metrics
                "min_width": {"$min": {"$arrayElemAt": ["$xywhn", 2]}},
                "max_width": {"$max": {"$arrayElemAt": ["$xywhn", 2]}},
                "min_height": {"$min": {"$arrayElemAt": ["$xywhn", 3]}},
                "max_height": {"$max": {"$arrayElemAt": ["$xywhn", 3]}},
                
                # Processing Time Metrics
                "avg_preprocess_time": {"$avg": "$speed.preprocess"},
                "avg_inference_time": {"$avg": "$speed.inference"},
                "avg_postprocess_time": {"$avg": "$speed.postprocess"},
                
                # Training Status
                "trained_count": {
                    "$sum": {
                        "$cond": [{"$eq": ["$is_trained", True]}, 1, 0]
                    }
                },
                "untrained_count": {
                    "$sum": {
                        "$cond": [{"$eq": ["$is_trained", False]}, 1, 0]
                    }
                },
                
                # Split Distribution
                "split_distribution": {
                    "$push": "$split"
                }
            }
        },
        
        # Stage 2: Sort by total count in descending order
        {"$sort": {"total_count": -1}},
        
        # Stage 3: Project to create a more readable output
        {
            "$project": {
                "class_name": "$_id",
                "_id": 0,
                "total_count": 1,
                "confidence_metrics": {
                    "average": "$avg_confidence",
                    "minimum": "$min_confidence", 
                    "maximum": "$max_confidence",
                    "standard_deviation": "$confidence_std_dev"
                },
                "size_metrics": {
                    "average_width": "$avg_width",
                    "average_height": "$avg_height", 
                    "average_area": "$avg_area",
                    "min_width": "$min_width",
                    "max_width": "$max_width",
                    "min_height": "$min_height", 
                    "max_height": "$max_height"
                },
                "processing_times": {
                    "avg_preprocess": "$avg_preprocess_time",
                    "avg_inference": "$avg_inference_time", 
                    "avg_postprocess": "$avg_postprocess_time"
                },
                "trained_count": "$trained_count",
                "untrained_count": "$untrained_count",
                "split_distribution": "$split_distribution"
            }
        }
    ]
    
    # Execute the aggregation pipeline
    results = list(collection.aggregate(pipeline))
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Additional analysis and transformations
    if not df.empty:
        # Expand nested dictionaries
        df['confidence_average'] = df['confidence_metrics'].apply(lambda x: x['average'])
        df['confidence_min'] = df['confidence_metrics'].apply(lambda x: x['minimum'])
        df['confidence_max'] = df['confidence_metrics'].apply(lambda x: x['maximum'])
        df['confidence_std'] = df['confidence_metrics'].apply(lambda x: x['standard_deviation'])
        
        # Calculate split distribution
        df['split_distribution'] = df['split_distribution'].apply(
            lambda splits: dict(pd.Series(splits).value_counts())
        )
    
    return df


def detailed_confidence_analysis(df):
    """
    Provide more detailed confidence analysis
    
    :param df: DataFrame from analyze_object_classes
    :return: DataFrame with additional confidence insights
    """
    confidence_analysis = df.copy()
    
    # Confidence buckets
    confidence_analysis['confidence_bucket'] = pd.cut(
        confidence_analysis['confidence_average'], 
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
        labels=['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    )
    
    return confidence_analysis

def visualize_class_metrics(df):
    """
    Create visualizations of class metrics
    
    :param df: DataFrame from analyze_object_classes
    :return: Matplotlib figure objects
    """
    import matplotlib.pyplot as plt
    
    # Create figure for class counts
    plt.figure(figsize=(10, 6))
    df.plot(kind='bar', x='class_name', y='total_count')
    plt.title('Object Class Counts')
    plt.xlabel('Class Name')
    plt.ylabel('Total Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Create figure for confidence metrics
    plt.figure(figsize=(10, 6))
    df.plot(kind='bar', x='class_name', y='confidence_average')
    plt.title('Average Confidence by Class')
    plt.xlabel('Class Name')
    plt.ylabel('Average Confidence')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# # Example usage
def main_cate():
    client = MongoInstance("traffic_analysis")
    client.select_collection("vehicle")
    
    # Perform analysis
    analysis_results = analyze_object_classes(client)
    
    # Display results
    print(analysis_results.info())
    
    # Optional: Save to CSV
    analysis_results.to_csv('object_class_analysis.csv', index=False)

def run_aggregation(query):
    results = list(client.aggregate(query))
    for result in results:
        print(result)

