import logging, os, ast
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import numpy as np

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

        self.model = YOLO(self.path, task=task )


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
                class_id = det.cls.item()
                if class_id in self.OBJECTS_OF_INTEREST.keys():
                    if class_id == 59: #number of face
                        unannotated_frame = self.face_blurring(unannotated_frame, det.xywh)
                        annotated_frame = self.face_blurring(annotated_frame, det.xywh)

                    output.append([
                        det.conf,
                        det.cls,
                        self.OBJECTS_OF_INTEREST[class_id],
                        det.xyxy,
                        det.xyxyn,
                        det.xywh,
                        det.xywhn,
                        frame_results.speed,
                        getattr(frame_results, "masks", None), #frame_results.masks,
                        frame_results.to_json(),
                    ])

                    # Annotate the frame
                    annotated_frame = self.annotate(
                        annotated_frame,
                        [self.OBJECTS_OF_INTEREST[class_id], det.xyxy[0], det.conf]
                    )

                    detected = True

            yield detected, unannotated_frame, annotated_frame, output 


    def annotate(self, frame, object_details: list, box_color: tuple = (54, 69, 79)):
        """
        Draws a bounding box and label on the frame for a detected object.

        Args:
            frame (numpy.ndarray): The image to annotate.
            object_details (list): List containing:
                * Label (str): Name of the object (e.g., 'person').
                * Bounding box (list or np.ndarray): Coordinates of the box in `[x_min, y_min, x_max, y_max]` format.
                * Confidence (float): Detection confidence score between 0 and 1.
            box_color (tuple, optional): RGB color for the bounding box. Defaults to (54, 69, 79).

        Returns:
            numpy.ndarray: Annotated image with bounding boxes and labels.
        """

        if len(object_details) != 3:
            raise ValueError("object_details must contain exactly three elements: [label, box, confidence].")

        label, box, confidence = object_details
        annotator = Annotator(frame, pil=False, line_width=1, example=label)
        formatted_label = f"{label}_{confidence.item() * 100:.2f}%"
        annotator.box_label(box=box, label=formatted_label, color=box_color)
        return annotator.result()


    def face_blurring(self, frame, xywh):
        """
        Applies a Gaussian blur to a specific region of the frame based on bounding box coordinates.

        Args:
            frame (numpy.ndarray): The input image (BGR format).
            xywh (list or np.ndarray): Bounding box in `[x_center, y_center, width, height]` format.

        Returns:
            numpy.ndarray: The image with the specified region blurred.
        """
        # Extract frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Convert xywh (relative format) to absolute pixel coordinates
        x_center, y_center, width, height = xywh
        x_center, y_center = int(x_center * frame_width), int(y_center * frame_height)
        width, height = int(width * frame_width), int(height * frame_height)

        # Calculate the bounding box corners
        x_min = max(x_center - width // 2, 0)
        y_min = max(y_center - height // 2, 0)
        x_max = min(x_center + width // 2, frame_width)
        y_max = min(y_center + height // 2, frame_height)

        # Apply Gaussian blur to the region
        blurred_frame = frame.copy()
        blurred_frame[y_min:y_max, x_min:x_max] = cv2.GaussianBlur(
            frame[y_min:y_max, x_min:x_max],
            (21, 21),
            0,
        )

        return blurred_frame
        
            







    