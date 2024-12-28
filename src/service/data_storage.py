import logging
import time
# from threading import Thread
from datetime import datetime
from db.mongo_instance import MongoInstance

logger = logging.getLogger(__name__)

class DataStorage:
    def __init__(self):
        self.client = MongoInstance("traffic_analysis")

    # @Thread
    def store_yearly_data(self, input_doc):
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
        self._store_data("yearly_metrics", input_doc, timestamp)

    # @Thread
    def store_monthly_data(self, input_doc):
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
        self._store_data("monthly_metrics", input_doc, timestamp)

    # @Thread
    def store_daily_data(self, input_doc):
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
        self._store_data("daily_metrics", input_doc, timestamp)

    # @Thread
    def store_category_data(self, input_doc):
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
        self._store_data("category_metrics", input_doc, timestamp)

    def _store_data(self, collection_name, input_doc, timestamp):
        self.client.select_collection(collection_name)
        logger.info(f"Storing {collection_name} data to the database.")
        try:
            start_time = time.process_time()
            input_doc["time_stamp"] = timestamp
            self.client.insert_data(input_doc)
            stop_time = time.process_time()
            logger.info(f"{collection_name.capitalize()} data storage complete, time taken: {stop_time - start_time}")
        except Exception as e:
            logger.error(f"Failed to store {collection_name} data: {str(e)}")