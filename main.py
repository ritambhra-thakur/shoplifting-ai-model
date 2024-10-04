import streamlit as st
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import cv2
import supervision as sv
from decouple import config
import os

# Initialize Supervision Annotator
annotator = sv.LabelAnnotator()

# Track if the alert has already been triggered
alert_triggered = False

# Function for prediction and annotations
def predi(predictions: dict, video_frame: VideoFrame):
    global alert_triggered  # Use the global variable to track the alert state
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_inference(predictions)
    image = annotator.annotate(
        scene=video_frame.image.copy(), detections=detections, labels=labels
    )
    cv2.imshow("Predictions", image)
    cv2.waitKey(1)

    if 'shoplifting' in labels or 'thief' in labels:
        if not alert_triggered:  # Only trigger warning if it hasn't been triggered yet
            st.warning("ALERT! Theft detected in the store", icon="🚨")
            alert_triggered = True

# Function to start the pipeline with the selected video
def start_pipeline(video_path):
    global alert_triggered
    st.write(f"Starting detection for video: {video_path}")
    alert_triggered = False  # Reset the alert trigger at the start of each pipeline run
    pipeline = InferencePipeline.init(
        model_id="theft-detection-using-cv-and-rl/2",
        api_key=config('roboflow_api_key'),
        video_reference=video_path,
        on_prediction=predi
    )
    pipeline.start()
    pipeline.join()
    cv2.destroyAllWindows()

# Streamlit UI starts here
st.title("Shoplifting Detection AI")

# Add a warning at the top of the app
st.warning("Caution: This system detects potential theft in real-time.")

# Get list of videos
video_folders = ["normal_videos", "shoplifting_videos"]
videos = []
for folder in video_folders:
    for video in os.listdir(folder):
        if video.endswith(('.mp4', '.webm')):
            videos.append(os.path.join(folder, video))

# Dropdown to select a video
selected_video = st.selectbox("Select a video to analyze", videos)

# Button to start analysis
if st.button("Start Detection"):
    if selected_video:
        start_pipeline(selected_video)
    else:
        st.error("Please select a video file to proceed.")
