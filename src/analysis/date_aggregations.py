from db.mongo_instance import MongoInstance

# Connect to MongoDB
client = MongoInstance("traffic_analysis")
client.select_collection("vehicle")

# Aggregation pipeline
# 1. Group by Month Query
month_query = [
    {
        '$addFields': {
            'parsed_month': {
                '$toInt': { '$substr': ['$dmy', 3, 2] }
            },
            'parsed_year': {
                '$toInt': { '$substr': ['$dmy', 6, 2] }
            }
        }
    },
    {
        '$group': {
            '_id': {
                'month': '$parsed_month',
                'year': { '$add': [2000, '$parsed_year'] }
            },
            'count': {'$sum': 1},
            'avg_confidence': {'$avg': '$confidence'},
            'classes': {'$addToSet': '$class_name'}
        }
    },
    {
        '$sort': {'_id.year': 1, '_id.month': 1}
    }
]

# 2. Group by Week Query
week_query = [
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
            }
        }
    },
    {
        '$group': {
            '_id': {
                'week': { '$week': '$parsed_date' },
                'year': { '$year': '$parsed_date' }
            },
            'count': {'$sum': 1},
            'avg_confidence': {'$avg': '$confidence'},
            'classes': {'$addToSet': '$class_name'}
        }
    },
    {
        '$sort': {'_id.year': 1, '_id.week': 1}
    }
]

# 3. Group by Day Query
day_query = [
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
            }
        }
    },
    {
        '$group': {
            '_id': {
                'day': { '$dayOfMonth': '$parsed_date' },
                'month': { '$month': '$parsed_date' },
                'year': { '$year': '$parsed_date' }
            },
            'count': {'$sum': 1},
            'avg_confidence': {'$avg': '$confidence'},
            'classes': {'$addToSet': '$class_name'}
        }
    },
    {
        '$sort': {
            '_id.year': 1, 
            '_id.month': 1, 
            '_id.day': 1
        }
    }
]

# 4. Group by Hour Query
hour_query = [
    {
        '$addFields': {
            'parsed_hour': {
                '$toInt': { '$substr': ['$time_stamp', 9, 2] }
            },
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
            }
        }
    },
    {
        '$group': {
            '_id': {
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
    {
        '$sort': {
            '_id.year': 1, 
            '_id.month': 1, 
            '_id.day': 1, 
            '_id.hour': 1
        }
    }
]

def run_aggregation(query):
    results = list(client.aggregate(query))
    for result in results:
        print(result)

def run_month_query():
    run_aggregation(month_query)

def run_week_query():
    run_aggregation(week_query)

def run_day_query():
    run_aggregation(day_query)

def run_hour_query():
    run_aggregation(hour_query)

def run_all_queries():
    print("Grouping by Month:")
    run_aggregation(month_query)
    
    print("Grouping by Week:")
    run_aggregation(week_query)
    
    print("Grouping by Day:")
    run_aggregation(day_query)
    
    print("Grouping by Hour:")
    run_aggregation(hour_query)

