import os, csv, shutil, time, threading, zipfile
from datetime import datetime
import cv2 as cv
from dotenv import load_dotenv, dotenv_values
from service.Transfer import Client
import shutil

client = Client()
load_dotenv()

def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

class DataService():

    def __init__(self) -> None:
        self.dmy = datetime.now().strftime("%d_%m_%y")
        self.OUT_DIR = os.getenv("OUT_DIR")
        self.ZIP_PLACE = os.getenv("ZIP_PLACE")

    def store_frame(self, frame: list, objects: list = None):
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
        filename = "unannotated_%s.png" % timestamp

        cv.imwrite("%s/unannotated_%s.png" % (self.OUT_DIR, timestamp), frame[0])
        cv.imwrite("%s/annotated_%s.png" % (self.OUT_DIR, timestamp), frame[1])

        return filename
        
    def store_metadata(self, objects: list = None, filename: str = None):
        fields = ["confidence", "class_id", "class_name", "data", "xmax", "ymax", "xmin", "ymin", "boxes", "time_stamp", "filename", "speed", "model_size"]
        path = '%s/traffic_%s.csv'% (self.OUT_DIR, self.dmy)
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
        
        #create the initial csv file
        if not os.path.exists(path):
            with open(path, 'w+') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fields)

        if objects:
            with open(path, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fields)

                for object in objects:                   
                    writer.writerow({"confidence": object.conf.item(),
                                     "class_id": object.cls_id.item(),
                                     "class_name": object.cls,
                                     "data": object.data,
                                     "xmax": object.coordinates[0].item(),
                                     "ymax": object.coordinates[1].item(),
                                     "ymin": object.coordinates[2].item(),
                                     "xmin": object.coordinates[3].item(),
                                     "speed": object.speed,
                                     "boxes": object.boxes,
                                     "time_stamp": timestamp,
                                     "filename": filename,
                                     "model_size": object.model_size})
                    
    def store_video(self, frame: list, objects: list = None):
        self.transfer_data()
        filename = self.store_frame(frame, objects)
        self.store_metadata(objects, filename)

    def _get_size(self):
        if os.path.exists(self.ZIP_PLACE):
            start_path = self.ZIP_PLACE
        else:
            start_path = self.OUT_DIR

        total_size = 0
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)

        return total_size

    @threaded
    def _zip_data(self, zip_path) -> bool:
        #PROBLEM: inputs the files in zip but they aren't removed after being put in zip
        """
        Zips all the .png files in the OUT_DIR directory into a zip file.

        Args:
            zip_path (str): The path to the zip file to be created.
        """
        with zipfile.ZipFile(zip_path, mode='w') as zfile:
            len_dir_path = len(self.OUT_DIR)
            for root, _, files in os.walk(self.OUT_DIR):
                for f in files:
                    if f.endswith(".png"):
                        zfile.write(os.path.join(root, f),  os.path.join(root, f)[len_dir_path:])
            return True
        return False
    
    def _remove_original(self):
        for root, _, files in os.walk(self.OUT_DIR):
            for f in files:
                if f.endswith(".png"):
                    os.remove(os.path.join(self.OUT_DIR, f))

    
            
    def transfer_data(self, zipped=False):
        """
        Transfers data to the server.
        
        Transfers all .png files in the OUT_DIR directory to the server as a zip file if the size of the directory is greater than 1.5 GB.
        Transfers the csv file to the server at midnight.
        """
        size = self._get_size()

        if size >= 0.5 * 1024 * 1024:
            print("[TRANSFER] Starting zipping")
            zipped = self._zip_data("/home/flor/Workspace/traffic_analysis/data/traffic_data.zip")
            print("[TRANSFER] Completed zipping")
            
            # Should wait till thread is completed before starting
        while not zipped:
            print("waiting till zip")
            time.sleep(1)
        
        if zipped:
            # self._remove_original()

            process = client.data_transfer("/home/flor/Workspace/traffic_analysis/data/traffic_data.zip")

            if not process:
                print("[TRANSFER] The transfer did not work")
                return
            
            print("[TRANSFER] Starting the transfer process")
        
        print("[TRANSFER] Not to big")

        if datetime.now().hour == 0 and datetime.now().minute == 0:
            client.data_transfer("%s/traffic_%s.csv"% self.OUT_DIR, self.dmy)
            print("[TRANSFER] CSV is being transfered")