from ultralytics import YOLO
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score, auc, mean_squared_error, accuracy_score, precision_recall_curve
import numpy as np
from db.mongo_instance import MongoInstance
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
from typing import List, Dict, Union, Tuple
import cv2

class ModelEvaluator:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, db_name: str = "traffic_analysis", collection_name: str = "evaluation_metrics"):
        """
        Initialize the YOLO model evaluator.
        
        Args:
            model_path (str): Path to the YOLO model weights
            confidence_threshold (float): Confidence threshold for predictions
        """
        self.model = YOLO(model_path, task="detect")
        self.confidence_threshold = confidence_threshold
        self.evaluation_results = {}
        self.predictions = []
        self.client = MongoInstance(db_name)
        self.client.select_collection(collection_name)


    def parse_coco_dataset(self, coco_file: str, split: str = 'test') -> Tuple[List[str], List[Dict]]:
        """
        Parse COCO dataset to extract image paths and ground truth for a specific split.
        
        Args:
            coco_file: Path to the COCO dataset JSON file.
            split: The split to filter images by (e.g., "test", "validation").
        
        Returns:
            Tuple containing a list of image paths and a list of ground truth annotations.
        """
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
        
        # Filter images based on split in the file name
        split_images = {
            img['id']: img['file_name']
            for img in coco_data['images']
            if split in img['file_name']
        }
        
        # Extract ground truth annotations only for the filtered images
        ground_truth = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id in split_images:
                if image_id not in ground_truth:
                    ground_truth[image_id] = {'boxes': [], 'classes': []}
                # Convert bbox from xywh to xyxy
                x, y, w, h = ann['bbox']
                ground_truth[image_id]['boxes'].append([x, y, x + w, y + h])
                for category in coco_data['categories']:
                    if category['id'] == ann['category_id']:
                        ground_truth[image_id]['classes'].append((category['id'], category['name']))
        
        # Prepare a list of ground truth dictionaries
        ground_truth_list = [
            {'boxes': ground_truth[img_id]['boxes'], 'classes': ground_truth[img_id]['classes']}
            for img_id in split_images.keys()
        ]
        
        return list(split_images.values()), ground_truth_list
        
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
        """
        metrics = {
            'per_class': {},
            'overall': {}
        }
        
        # Get unique classes
        all_classes = set()
        for gt in ground_truth:
            all_classes.update(gt['classes'])
        
        # Calculate per-class metrics
        overall_true = []
        overall_pred = []
        overall_scores = []
        
        for class_id, class_name in all_classes:
            y_true = []
            y_pred = []
            y_scores = []
            
            for gt, pred in zip(ground_truth, predictions):
                gt_mask = np.array([cls[0] for cls in gt['classes']]) == class_id
                pred_mask = np.array(pred['classes']) == class_id
                print(f"this is gt_mask: {gt_mask}\n this is pred_mask: {pred_mask}")

                # Add ground truth
                for _ in range(sum(gt_mask)):
                    if sum(pred_mask) > 0:
                        y_true.append(1)
                        y_pred.append(1)
                        y_scores.append(max(pred['confidence'][pred_mask]))
                    else:
                        y_true.append(1)
                        y_pred.append(0)
                        y_scores.append(0)
                
                # # Add false positives
                # for conf in pred['confidence'][pred_mask][sum(gt_mask):]:
                #     y_true.append(0)
                #     y_pred.append(1)
                #     y_scores.append(conf)
            
            if len(y_true) > 0:
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                
                # Calculate precision-recall curve for AUC
                precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
                auc_score = auc(recalls, precisions)
                
                # metrics['per_class'][str(class_name)] = {
                #     'accuracy': accuracy_score(y_true, y_pred),
                #     'mean_ap': self._calculate_map(y_true, y_pred),
                #     'average_iou': self._calculate_average_iou(y_true, y_pred),
                #     'mean_squared_error': mean_squared_error(y_true, y_pred),
                #     'auc': auc_score(precisions, recalls),
                #     'precision': precision,
                #     'recall': recall,
                #     'f1': f1,
                #     'auc': auc_score
                # }
                
                overall_true.extend(y_true)
                overall_pred.extend(y_pred)
                overall_scores.extend(y_scores)
        
        # Calculate overall metrics
        overall_precision = precision_score(overall_true, overall_pred)
        overall_recall = recall_score(overall_true, overall_pred)
        overall_f1 = f1_score(overall_true, overall_pred)
        
        # Calculate overall precision-recall curve for AUC
        precisions, recalls, _ = precision_recall_curve(overall_true, overall_scores)
        overall_auc = auc(recalls, precisions)
        
        metrics['overall'] = {
            'accuracy': accuracy_score(overall_true, overall_pred),
            'mean_ap': self._calculate_map(ground_truth, predictions),
            'average_iou': self._calculate_average_iou(ground_truth, predictions),
            'mean_squared_error': mean_squared_error(overall_true, overall_pred),
            'auc': overall_auc,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1
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
            matched_classes = self._match_boxes(gt['boxes'], pred['boxes'], [cls[0] for cls in gt['classes']], pred['classes'])
            y_true.extend([cls[0] for cls in gt['classes']])
            y_pred.extend(matched_classes)
            
        return np.array(y_true), np.array(y_pred)
    

    def _match_boxes(self, gt_boxes: List[np.ndarray], pred_boxes: List[np.ndarray], gt_classes: List[int], pred_classes: List[int]) -> List[int]:
        """
        Match predicted boxes to ground truth boxes using IoU.
        
        Args:
            gt_boxes: List of ground truth bounding boxes
            pred_boxes: List of predicted bounding boxes
            gt_classes: List of ground truth classes
            pred_classes: List of predicted classes
            
        Returns:
            List of matched predicted classes
        """
        matched_classes = [-1] * len(gt_boxes)
        for i, gt_box in enumerate(gt_boxes):
            ious = [self._calculate_iou(gt_box, pred_box) for pred_box in pred_boxes]
            if ious:
                max_iou_idx = np.argmax(ious)
                if ious[max_iou_idx] > 0.5:  # IoU threshold for matching
                    matched_classes[i] = pred_classes[max_iou_idx]
        return matched_classes


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
        """
        aps = []
        print(ground_truth)
        for class_id in np.unique(np.concatenate([[cls[0] for cls in gt['classes']] for gt in ground_truth])):
            y_true = []
            y_scores = []
            
            for gt, pred in zip(ground_truth, predictions):
                gt_mask = np.array([cls[0] for cls in gt['classes']]) == class_id
                pred_mask = np.array(pred['classes']) == class_id
                
                # For each ground truth box of this class
                for _ in range(sum(gt_mask)):
                    if sum(pred_mask) > 0:
                        # Add the highest confidence prediction
                        y_true.append(1)
                        y_scores.append(max(pred['confidence'][pred_mask]))
                    else:
                        # No prediction for this ground truth
                        y_true.append(1)
                        y_scores.append(0)
                
                # Add false positives
                for conf in pred['confidence'][pred_mask][sum(gt_mask):]:
                    y_true.append(0)
                    y_scores.append(conf)
            
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

        def _convert_numpy(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: _convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [_convert_numpy(item) for item in obj]
            return obj
                
        results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'confidence_threshold': self.confidence_threshold,
            'metrics': _convert_numpy(self.evaluation_results)
        }

        self.client.insert_data(results)
        
            
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