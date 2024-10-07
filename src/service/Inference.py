import numpy as np
import cv2 as cv
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

class Inference():

    def __init__(self, path: str = "/home/flor/Workspace/traffic_analysis/models/yolov8s_ncnn_model", task: str = "detect", confidence: float = 0.75):

        self.OBJECTS_OF_INTEREST = {0:"person", 1:"bicycle", 2:"car", 5:"bus", 7: "truck", 14:"bird", 15:"cat", 16:"dog"}
        self.confidence = confidence

        if not path:
            self.model = YOLO("yolov8s.pt")
            self.model.export(format="ncnn")
            self.model = YOLO("yolov8s_ncnn_model", task="detect")
        else:
            self.model = YOLO(path, task="detect")
    
    def detect(self, frame):
        results  = self.model(frame)

        detected = False
        output = []
        unannotated_frame = frame
        annotated_frame = None

        for frame_results in results:
            for det in frame_results.boxes:
                if det.cls.item() in self.OBJECTS_OF_INTEREST.keys() and det.conf >= self.confidence: 
                    output.append(DetectedObject(det.xyxy[0],
                                                 det.conf,
                                                 det.cls,
                                                 self.OBJECTS_OF_INTEREST[det.cls.item()],
                                                 det.data,
                                                 det.xywh,
                                                 frame_results.speed,
                                                 self.model.model))
                    
                    annotated_frame = self.annotate(unannotated_frame, [self.OBJECTS_OF_INTEREST[det.cls.item()], det.xyxy[0], det.conf])

                    detected = True
        
        return [detected, unannotated_frame, annotated_frame, output]
    
    def annotate(self, frame, objects: list, box_color: tuple = (227, 16, 44)):
        annotator = Annotator(frame, pil=False, line_width=2, example=objects[0])
        annotator.box_label(box=objects[1], label=f"{objects[0]}_{(objects[2].item()*100):.2f}", color=box_color)
        return annotator.result()

class DetectedObject():

    def __init__(self, coordinates: list, conf: float, cls_id: int, cls:str, data: list, boxes, speed, model) -> None:
        self.coordinates = coordinates
        self.conf = conf
        self.cls_id = cls_id
        self.cls = cls
        self.data = data
        self.boxes = boxes
        self.speed = speed
        self.model_size = model.strip

    def __repr__(self) -> str:
        return f"Box coordinates: {self.coordinates[0]}, {self.coordinates[1]}, {self.coordinates[2]}, {self.coordinates[3]}, \nConfidence: {self.conf}, \nClass id: {self.cls_id}, \nClass: {self.cls}"
        

        
            







    