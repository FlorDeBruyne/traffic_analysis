import logging
from utils.capture_utils import main
from service.image_processing import ImageProcessor


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    processor = ImageProcessor()
    processor.process_dataset(
        output_dir="../output_images",
        coco_output_file="../coco_dataset.json",
        untrained_only=True
    )
 