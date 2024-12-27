import os
from service.webcam_controller import WebcamController


def run_streamlit():
    """
    Run the Streamlit app from within the Capture file.
    """
    os.system("streamlit run /home/flor/traffic_analysis/src/dashboard/dashboard.py")

def main():
    run_streamlit()
    # webcam = WebcamController()
    # webcam.camera_setup()
    # webcam.stream_video()

if __name__ == "__main__":
    main()

