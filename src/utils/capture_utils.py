import os
from service.webcam_controller import WebcamController


def run_streamlit():
    """
    Run the Streamlit app from within the Capture file.
    """
    os.system("streamlit run stream_webpage.py --server.address 0.0.0.0 --server.port 8501")

def main():
    webcam = WebcamController()
    webcam.camera_setup()
    logging.basicConfig(filename='logs/info_capture.log', level=logging.INFO)
    logger.info("Started")
    webcam.stream_video()

if __name__ == "__main__":
    main()

