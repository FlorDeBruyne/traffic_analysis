from ultralytics import YOLO
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, auc, mean_squared_error, accuracy_score


class ModelEvaluator:
    def __init__(self, model_path, confidence_threshold=0.25):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def predict(self, images: list):
        pass

    def evaluate_model(self, ground_thruth: list, predictions: list):
        pass

    def save_evaluations(self):
        pass

    def visualize_predictions(self):
        pass