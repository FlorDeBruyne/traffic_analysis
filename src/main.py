import logging
from utils.capture_utils import main
from service.image_processing import ImageProcessor
from model_utils.evaluate_model import ModelEvaluator

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='logs/info_capture.log', level=logging.INFO)
    logger.info("Started")


    # evaluator = ModelEvaluator(model_path="/home/flor/traffic_analysis/models/objectdetection/yolov8n_ncnn_model", confidence_threshold=0.5)
    # coco_file = "/home/flor/traffic_analysis/coco_dataset.json"
    
    # image_paths, ground_truth = evaluator.parse_coco_dataset(coco_file, 'test')

    # predictions = evaluator.predict(image_paths)

    # metrics = evaluator.evaluate_model(ground_truth, predictions)
    # print("saving the evaulation")
    # evaluator.save_evaluations()
    # print(metrics)
    main()


    # processor = ImageProcessor()
    # processor.process_dataset(
    #     output_dir="../output_images",
    #     coco_output_file="../coco_dataset.json",
    #     untrained_only=True
    # )
 