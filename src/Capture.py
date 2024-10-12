
import logging
import numpy as np
from service.WebcamController import WebcamController
logger = logging.getLogger(__name__)

webcam = WebcamController()
logging.basicConfig(filename='capture.log', level=logging.INFO)
logger.info("Started")
webcam.stream_video()


