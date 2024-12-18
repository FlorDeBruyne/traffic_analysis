from db.mongo_instance import MongoInstance
import pandas as pd

# Connect to MongoDB
client = MongoInstance("traffic_analysis")
client.select_collection("vehicle")

def analyse_object_dates(collection):
    combined_query = [
        # Initial date parsing stage
        {
            '$addFields': {
                'parsed_date': {
                    '$dateFromString': {
                        'dateString': {
                            '$concat': [
                                {'$substr': ['$time_stamp', 0, 2]},  # Day
                                '/',
                                {'$substr': ['$time_stamp', 3, 2]},  # Month
                                '/',
                                {'$concat': ['20', {'$substr': ['$time_stamp', 6, 2]}]}  # Year
                            ]
                        },
                        'format': '%d/%m/%Y'
                    }
                },
                'parsed_hour': {
                    '$toInt': { '$substr': ['$time_stamp', 9, 2] }
                }
            }
        },
        
        # Month Aggregation
        {
            '$group': {
                '_id': {
                    'granularity': 'month',
                    'month': { '$month': '$parsed_date' },
                    'year': { '$year': '$parsed_date' }
                },
                'count': {'$sum': 1},
                'avg_confidence': {'$avg': '$confidence'},
                'classes': {'$addToSet': '$class_name'}
            }
        },
        
        # Week Aggregation
        {
            '$group': {
                '_id': {
                    'granularity': 'week',
                    'week': { '$week': '$parsed_date' },
                    'year': { '$year': '$parsed_date' }
                },
                'count': {'$sum': 1},
                'avg_confidence': {'$avg': '$confidence'},
                'classes': {'$addToSet': '$class_name'}
            }
        },
        
        # Day Aggregation
        {
            '$group': {
                '_id': {
                    'granularity': 'day',
                    'day': { '$dayOfMonth': '$parsed_date' },
                    'month': { '$month': '$parsed_date' },
                    'year': { '$year': '$parsed_date' }
                },
                'count': {'$sum': 1},
                'avg_confidence': {'$avg': '$confidence'},
                'classes': {'$addToSet': '$class_name'}
            }
        },
        
        # Hour Aggregation
        {
            '$group': {
                '_id': {
                    'granularity': 'hour',
                    'hour': '$parsed_hour',
                    'day': { '$dayOfMonth': '$parsed_date' },
                    'month': { '$month': '$parsed_date' },
                    'year': { '$year': '$parsed_date' }
                },
                'count': {'$sum': 1},
                'avg_confidence': {'$avg': '$confidence'},
                'classes': {'$addToSet': '$class_name'}
            }
        },
        
        # Sorting stage
        {
            '$sort': {
                '_id.granularity': 1,
                '_id.year': 1,
                '_id.month': 1,
                '_id.day': 1,
                '_id.hour': 1
            }
        }
    ]

    results = list(collection.aggregate(combined_query))
    
    # Organize results by granularity
    aggregations = {
        'month': [],
        'week': [],
        'day': [],
        'hour': []
    }
    
    for result in results:
        granularity = result['_id']['granularity']
        aggregations[granularity].append(result)
    
    return results

def main_date():
    client = MongoInstance("traffic_analysis")
    client.select_collection("vehicle")
    
    # Perform analysis
    analysis_results = analyse_object_dates(client)
    
    # Display results
    print(analysis_results)
    
    # Optional: Save to CSV
    # analysis_results.to_csv('object_date_analysis.csv', index=False)



def transform_aggregations_for_auto_report(aggregations):
    """
    Transform aggregation results into a format suitable for auto_report collection
    
    :param aggregations: Dictionary of aggregation results
    :return: List of documents for auto_report collection
    """
    auto_report_docs = []
    
    for granularity, results in aggregations.items():
        for result in results:
            doc = {
                "type": "temporal_aggregation",
                "time_granularity": granularity,
                "time_details": result['_id'],
                "detection_summary": {
                    "total_count": result['count'],
                    "avg_confidence": result['avg_confidence'],
                    "detected_classes": result['classes']
                }
            }
            auto_report_docs.append(doc)
    
    return auto_report_docs


