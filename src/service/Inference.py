import logging, os, ast
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


logger = logging.getLogger(__name__)

class Inference():
    def __init__(self, task: str = "detect", confidence: float = 0.7, size: str ='nano'):

        self.OBJECTS_OF_INTEREST = ast.literal_eval(os.getenv("OBJECTS_INTREST"))
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
        """
        Process a frame to detect objects of interest and annotate them.

        Args:
            frame (numpy.ndarray): The input frame from the video stream.

        Yields:
            tuple: (detected, unannotated_frame, annotated_frame, output)
                - detected (bool): Whether an object of interest was detected.
                - unannotated_frame (numpy.ndarray): Original frame.
                - annotated_frame (numpy.ndarray): Frame with annotations.
                - output (list): Details of the detected objects.
        """
        results = self.model(frame, stream=True, imgsz=640, conf=self.confidence)

        detected = False
        output = []
        unannotated_frame = frame.copy()
        annotated_frame = frame.copy()

        for frame_results in results:
            for det in frame_results.boxes:
                if det.cls.item() in self.OBJECTS_OF_INTEREST.keys():
                    output.append([
                        det.conf,
                        det.cls,
                        self.OBJECTS_OF_INTEREST[det.cls.item()],
                        det.xyxy,
                        det.xyxyn,
                        det.xywh,
                        det.xywhn,
                        frame_results.speed,
                        frame_results.masks,
                        frame_results.tojson(),
                    ])

                    # Annotate the frame
                    annotated_frame = self.annotate(
                        unannotated_frame,
                        [self.OBJECTS_OF_INTEREST[det.cls.item()], det.xyxy[0], det.conf]
                    )

                    detected = True

            yield detected, unannotated_frame, annotated_frame, output 
    
    
    def annotate(self, frame, objects: list, box_color: tuple = (227, 16, 44)):
        annotator = Annotator(frame, pil=False, line_width=2, example=objects[0])
        annotator.box_label(box=objects[1], label=f"{objects[0]}_{(objects[2].item()*100):.2f}", color=box_color)
        return annotator.result()

    def face_blurring(self, frame, mask):
        pass

        
            







    