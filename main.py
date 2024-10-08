import streamlit as st
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import cv2
import supervision as sv
from decouple import config
import os
from twilio.rest import Client  

# Annotator and Twilio setup
annotator = sv.LabelAnnotator()
alert_triggered = False
default_recipient_phone_number = config('RECIPIENT_PHONE_NUMBER')

client = Client(config('TWILIO_ACCOUNT_SID'), config('TWILIO_AUTH_TOKEN'))

def send_sms_alert(recipient_phone_number):
    message = client.messages.create(
        body="ALERT! Theft detected in the store.(testing 1)",
        from_=config('TWILIO_PHONE_NUMBER'),
        to=recipient_phone_number
    )
    st.write(f"SMS alert sent to {recipient_phone_number}: {message.sid}")

def predi(predictions: dict, video_frame: VideoFrame, recipient_phone_number):
    global alert_triggered
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_inference(predictions)
    image = annotator.annotate(
        scene=video_frame.image.copy(), detections=detections, labels=labels
    )
    # cv2.imshow("Predictions", image)
    cv2.waitKey(1)

    if 'shoplifting' in labels or 'thief' in labels:
        if not alert_triggered: 
            st.warning("ALERT! Theft detected in the store", icon="🚨")
            alert_triggered = True
            send_sms_alert(recipient_phone_number)
            

def start_pipeline(video_path, recipient_phone_number):
    global alert_triggered
    st.write(f"Starting detection for video: {video_path}")
    alert_triggered = False
    pipeline = InferencePipeline.init(
        model_id="theft-detection-using-cv-and-rl/2",
        api_key=config('roboflow_api_key'),
        video_reference=video_path,
        on_prediction=lambda predictions, video_frame: predi(predictions, video_frame, recipient_phone_number)
    )
    pipeline.start()
    pipeline.join()
    cv2.destroyAllWindows()
    st.success("Video Analyzed Successfully.")

# App title
st.title("Shoplifting Detection AI")

# Alert message
st.warning("Caution: This system detects potential theft in real-time.")

# User input for phone number
user_input_recipient_phone_number = st.text_input(f"Enter recipient phone number for alerts (leave blank to use default - {default_recipient_phone_number})")
recipient_phone_number = user_input_recipient_phone_number if user_input_recipient_phone_number else default_recipient_phone_number

# List available videos
video_folders = ["normal_videos", "shoplifting_videos"]
videos = []
for folder in video_folders:
    for video in os.listdir(folder):
        if video.endswith(('.mp4', '.webm')):
            videos.append(os.path.join(folder, video))

# Select video
selected_video = st.selectbox("Select a video to analyze", videos)

# Video player to play the selected video
if selected_video:
    st.video(selected_video)

# Rerun the app if a new video is selected
if 'previous_video' not in st.session_state:
    st.session_state['previous_video'] = selected_video

if st.session_state['previous_video'] != selected_video:
    st.session_state['previous_video'] = selected_video
    st.rerun()

# Start detection on button press
if st.button("Start Detection"):
    if selected_video:
        start_pipeline(selected_video, recipient_phone_number)
    else:
        st.error("Please select a video file to proceed.")
