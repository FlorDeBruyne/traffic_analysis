import logging
import os
from service.WebcamController import WebcamController


def run_streamlit():
    """
    Run the Streamlit app from within the Capture file.
    """
    os.system("streamlit run stream_webpage.py --server.address 0.0.0.0 --server.port 8501")

if __name__ == "__main__":
    
    logger = logging.getLogger(__name__)

    # webcam = WebcamController()
    logging.basicConfig(filename='capture.log', level=logging.INFO)
    logger.info("Started")
    # webcam.stream_video()
    run_streamlit()
