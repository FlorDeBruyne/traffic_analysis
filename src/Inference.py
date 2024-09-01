import numpy as np
import cv2 as cv
from ultralytics import YOLO

class Inference():

    def __init__(self, path: str = "../models/yolov8n_ncnn_model", task: str = "detect", confidence: float = 0.5):

        self.OBJECTS_OF_INTEREST = {0:"person", 1:"bicycle", 2:"car", 5:"bus", 14:"bird", 15:"cat", 16:"dog"}
        self.confidence = confidence

        if not path:
            self.model = YOLO("yolov8s.pt")
            self.model.export(format="ncnn")
            self.model = YOLO("yolov8s_ncnn_model", task="detect")
        else:
            self.model = YOLO(path, task="detect")
    
    def detect(self, frame):
        results  = self.model(frame, imgsz=(736, 960), conf=self.confidence)

        detected = False
        output = []
        output_frames = []

        for frame_results in results:
            for det in frame_results.boxes:
                if det.cls.item() in self.OBJECTS_OF_INTEREST.keys():
                    output.append(DetectedObject(det.xyxy[0], det.conf, det.cls, self.OBJECTS_OF_INTEREST[det.cls.item()], det.data))
                    output_frames.append(det)
                    detected = True

        return [detected, output_frames, output]

class DetectedObject():

    def __init__(self, coordinates: list, conf: float, cls_id: int, cls:str, data: list) -> None:
        self.coordinatess = coordinates
        self.conf = conf
        self.cls_id = cls_id
        self.cls = cls
        self.data = data

    def __repr__(self) -> str:
        return f"Box coordinates: {self.coordinatess[0]}, {self.coordinatess[1]}, {self.coordinatess[2]}, {self.coordinatess[3]}, \nConfidence: {self.conf}, \nClass id: {self.cls_id}, \nClass: {self.cls}"
        

        
            







    