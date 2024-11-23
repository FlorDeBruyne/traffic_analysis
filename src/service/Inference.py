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
            
            
            if frame_results is not None:
                # logger.info(f"This is a frame_result: {frame_results}")
                for item in frame_results:
                    logger.info(f"this is what is inside a frame_results: {item}\n")
                    for i in item:
                        logger.info(f"this is what is inside {item}: \n {i}\n")

                # if 'person' in frame_results.boxes.cls:
                #     frame_results.orig_img = self.face_blurring(frame_results.orig_img, frame_results.masks)
                
                
                # output.append([frame_results.boxes.xywh[0],
                #                frame_results.boxes.xywhn[0],
                #                frame_results.boxes.xyxy[0],
                #                frame_results.boxes.xyxyn[0],
                #                 frame_results.keypoints,
                #                 frame_results.masks,
                #     z            frame_results.obb, #LIST???
                #                 frame_results.orig_img,
                #                 frame_results.probs,
                #                 frame_results.speed[0], # preprocess
                #                 frame_results.speed[1], # inference
                #                 frame_results.speed[2], # postprocess
                #                 self.model.model])
                
            #     print("Output appended")

            #     annotated_frame = self.annotate(unannotated_frame.copy(), [self.OBJECTS_OF_INTEREST[frame_results.cls.item()], frame_results.xyxy[0], frame_results.conf])

            #     detected = True
            # print("No object")
        
        return None #[detected, unannotated_frame, annotated_frame, output]
    
    def annotate(self, frame, objects: list, box_color: tuple = (227, 16, 44)):
        annotator = Annotator(frame, pil=False, line_width=2, example=objects[0])
        annotator.box_label(box=objects[1], label=f"{objects[0]}_{(objects[2].item()*100):.2f}", color=box_color)
        return annotator.result()

    def face_blurring(self, frame, mask):
        """
        Input: A frame with a person in it, mask
        Output: A frame with the face of the person blurred

        use gaussian blur to blur out the face of the person in the frame.
        """
        print("Face should be blurred")
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
        

        
            







    