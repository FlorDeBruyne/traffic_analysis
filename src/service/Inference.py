import numpy as np
import cv2 as cv
import logging, threading,os 
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator




logger = logging.getLogger(__name__)

class Inference():
    def __init__(self, task: str = "segment", confidence: float = 0.7, size: str ='nano'):

        self.OBJECTS_OF_INTEREST = os.getenv("OBJECTS_INTREST")
        self.confidence = confidence

        if task == "segment":
            if size == "nano":
                self.path = os.getenv("SEG_NANO")
            else:
                self.path = os.getenv("SEG_SMALL")
        else:
            if size == "nano":
                self.path = os.getenv("DET_NANO")
            else:
                self.path = os.getenv("DET_SMALL")

        self.model = YOLO(self.path, task=task, )


    def detect(self, frame):
        results = self.model(frame, stream=True, imgsz=640, conf=self.confidence)

        detected = False
        output = []
        unannotated_frame = frame
        annotated_frame = None

        for frame_results in results:
            # print(f"This is a frame result:{frame_results} in results")
            logger.info(f"This is a frame_results in results: {frame_results}\n")
            
            logger.info(f"This is a boxes from frame_results: {frame_results.boxes}")
            
            
            logger.info(f"This is a masks from frame_results: {frame_results.masks}")
            
            if frame_results.masks != None:
                output.append([frame_results.boxes,
                                frame_results.keypoints,
                                frame_results.masks,
                                frame_results.obb,
                                frame_results.orig_img,
                                frame_results.probs,
                                frame_results.speed])

            # Boxes attributes:
            # cls
            # conf
            # id ????
            # xywh
            # xywhn
            # xyxy
            # xyxyn



                # if det.cls.item() in self.OBJECTS_OF_INTEREST.keys() and det.conf >= self.confidence: 
                #     output.append(DetectedObject(det.xyxy[0],
                #                                  det.conf,
                #                                  det.cls,
                #                                  self.OBJECTS_OF_INTEREST[det.cls.item()],
                #                                  det.data,
                #                                  det.xywh,
                #                                  frame_results.speed,
                #                                  self.model.model,
                #                                  frame_results.masks))
                    
                # annotated_frame = self.annotate(unannotated_frame, [self.OBJECTS_OF_INTEREST[det.cls.item()], det.xyxy[0], det.conf])

                # detected = True
        
        return [detected, unannotated_frame, annotated_frame, output]
    
    def annotate(self, frame, objects: list, box_color: tuple = (227, 16, 44)):
        annotator = Annotator(frame, pil=False, line_width=2, example=objects[0])
        annotator.box_label(box=objects[1], label=f"{objects[0]}_{(objects[2].item()*100):.2f}", color=box_color)
        return annotator.result()

    def face_blurring(self, frame, mask):
        pass

# class DetectedObject():

#     def __init__(self, coordinates: list, conf: float, cls_id: int, cls:str, data: list, boxes, speed, model, masks) -> None:
#         self.coordinates = coordinates
#         self.conf = conf
#         self.cls_id = cls_id
#         self.cls = cls
#         self.data = data
#         self.boxes = boxes
#         self.speed = speed
#         self.model = model
#         self.masks = masks

#     def __repr__(self) -> str:
#         return f"Box coordinates: {self.coordinates[0]}, {self.coordinates[1]}, {self.coordinates[2]}, {self.coordinates[3]}, \nConfidence: {self.conf}, \nClass id: {self.cls_id}, \nClass: {self.cls}"
        

        
            







    