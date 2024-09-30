from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import cv2
# import supervision to help visualize our predictions
import supervision as sv

annotator = sv.LabelAnnotator()


def predi(predictions: dict, video_frame: VideoFrame):
    print(predictions)
    print("Yeahhhh!!!")
    labels = [p["class"] for p in predictions["predictions"]]
    # load our predictions into the Supervision Detections api
    detections = sv.Detections.from_inference(predictions)
    # annotate the frame using our supervision annotator, the video_frame, the predictions (as supervision Detections), and the prediction labels
    image = annotator.annotate(
        scene=video_frame.image.copy(), detections=detections, labels=labels
    )
    # display the annotated image
    cv2.imshow("Predictions", image)
    cv2.waitKey(1)




# Use webcam by setting video_reference to 0
pipeline = InferencePipeline.init(
    model_id="shoplifting-detection-oxvwp/1",
    video_reference=0,  # This accesses the default webcam
    on_prediction=predi
)

pipeline.start()
pipeline.join()
