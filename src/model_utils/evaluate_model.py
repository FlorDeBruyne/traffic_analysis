from ultralytics import YOLO
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, auc, mean_squared_error, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
from typing import List, Dict, Union, Tuple
import cv2

class ModelEvaluator:
    def __init__(self, model_path: str, confidence_threshold: float = 0.25):
        """
        Initialize the YOLO model evaluator.
        
        Args:
            model_path (str): Path to the YOLO model weights
            confidence_threshold (float): Confidence threshold for predictions
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.evaluation_results = {}
        self.predictions = []
        
    def predict(self, images: List[Union[str, np.ndarray]]) -> List[Dict]:
        """
        Make predictions on a list of images.
        
        Args:
            images: List of image paths or numpy arrays
            
        Returns:
            List of predictions for each image
        """
        results = []
        for img in images:
            prediction = self.model(img, conf=self.confidence_threshold)
            processed_pred = self._process_prediction(prediction[0])
            results.append(processed_pred)
            self.predictions.append(processed_pred)
        return results
    
    def _process_prediction(self, prediction) -> Dict:
        """
        Process raw YOLO prediction into a structured format.
        
        Args:
            prediction: Raw YOLO prediction
            
        Returns:
            Processed prediction dictionary
        """
        boxes = prediction.boxes
        processed = {
            'boxes': boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else [],
            'confidence': boxes.conf.cpu().numpy() if boxes.conf is not None else [],
            'classes': boxes.cls.cpu().numpy() if boxes.cls is not None else []
        }
        return processed
    
    def evaluate_model(self, ground_truth: List[Dict], predictions: List[Dict]) -> Dict:
        """
        Evaluate model performance against ground truth.
        
        Args:
            ground_truth: List of ground truth annotations
            predictions: List of model predictions
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Initialize metrics storage
        metrics = {
            'per_class': {},
            'overall': {}
        }
        
        # Calculate per-class metrics
        y_true, y_pred = self._prepare_data_for_metrics(ground_truth, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
        
        # Calculate overall metrics
        metrics['overall'] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'mean_ap': self._calculate_map(ground_truth, predictions),
            'average_iou': self._calculate_average_iou(ground_truth, predictions),
            'mean_squared_error': mean_squared_error(y_true, y_pred),
            'auc': auc(recall, precision),
            'precision': np.mean(precision),
            'recall': np.mean(recall),
            'f1': np.mean(f1),
        }
        
        self.evaluation_results = metrics
        return metrics
    
    def _prepare_data_for_metrics(self, ground_truth: List[Dict], predictions: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare ground truth and predictions for metric calculation.
        
        Args:
            ground_truth: List of ground truth annotations
            predictions: List of model predictions
            
        Returns:
            Tuple of ground truth and prediction arrays
        """
        y_true = []
        y_pred = []
        
        for gt, pred in zip(ground_truth, predictions):
            # Match predictions to ground truth using IoU
            matched_classes = self._match_boxes(gt['boxes'], pred['boxes'], gt['classes'], pred['classes'])
            y_true.extend(gt['classes'])
            y_pred.extend(matched_classes)
            
        return np.array(y_true), np.array(y_pred)
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes.
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU score
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        iou = intersection / (box1_area + box2_area - intersection + 1e-6)
        return iou
    
    def _calculate_average_iou(self, ground_truth: List[Dict], predictions: List[Dict]) -> float:
        """
        Calculate average IoU across all predictions.
        
        Args:
            ground_truth: List of ground truth annotations
            predictions: List of model predictions
            
        Returns:
            Average IoU score
        """
        total_iou = 0
        total_boxes = 0
        
        for gt, pred in zip(ground_truth, predictions):
            if len(gt['boxes']) == 0 or len(pred['boxes']) == 0:
                continue
                
            for gt_box in gt['boxes']:
                ious = [self._calculate_iou(gt_box, pred_box) for pred_box in pred['boxes']]
                if ious:
                    total_iou += max(ious)
                    total_boxes += 1
                    
        return total_iou / total_boxes if total_boxes > 0 else 0
    
    def _calculate_map(self, ground_truth: List[Dict], predictions: List[Dict]) -> float:
        """
        Calculate mean Average Precision.
        
        Args:
            ground_truth: List of ground truth annotations
            predictions: List of model predictions
            
        Returns:
            mAP score
        """
        # Simplified mAP calculation
        aps = []
        for class_id in np.unique(np.concatenate([gt['classes'] for gt in ground_truth])):
            y_true = []
            y_scores = []
            
            for gt, pred in zip(ground_truth, predictions):
                gt_mask = gt['classes'] == class_id
                pred_mask = pred['classes'] == class_id
                
                y_true.extend([1] * sum(gt_mask))
                y_true.extend([0] * (sum(pred_mask) - sum(gt_mask)))
                
                y_scores.extend(pred['confidence'][pred_mask])
            
            if len(y_true) > 0 and len(y_scores) > 0:
                ap = average_precision_score(y_true, y_scores)
                aps.append(ap)
                
        return np.mean(aps) if aps else 0
    
    def save_evaluations(self, save_path: str = "evaluation_results.json"):
        """
        Save evaluation results to a JSON file.
        
        Args:
            save_path: Path to save the evaluation results
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results to save. Run evaluate_model first.")
            
        results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'confidence_threshold': self.confidence_threshold,
            'metrics': self.evaluation_results
        }
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
            
    def visualize_predictions(self, image_path: str, prediction: Dict, save_path: str = None):
        """
        Visualize predictions on an image.
        
        Args:
            image_path: Path to the image
            prediction: Prediction dictionary for the image
            save_path: Optional path to save the visualization
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw bounding boxes
        for box, conf, cls in zip(prediction['boxes'], prediction['confidence'], prediction['classes']):
            x1, y1, x2, y2 = map(int, box)
            color = (255, 0, 0)  # Red color for boxes
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"Class {int(cls)} ({conf:.2f})"
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()