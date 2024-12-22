from utils.capture_utils import main
from service.image_processing import ImageProcessor


if __name__ == "__main__":
    # Example usage
    processor = ImageProcessor()
    processor.extract_untrained_images(output_dir="../output_images", coco_output_file="coco_dataset.json")




    # main()