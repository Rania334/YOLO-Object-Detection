
# YOLO-Object-Detection
Real-time object detection using YOLOv3 and OpenCV. Detects and labels objects in video streams with bounding boxes and confidence scores. Features include optional GPU acceleration and the ability to log detections and save output video.


## Features
- Real-time object detection from webcam or video files.
- Detection and labeling of multiple objects with bounding boxes.
- Displays confidence scores for detected objects.
- Optional GPU acceleration for improved performance.
- Saves detected objects' information in JSON format.
- Saves the output video with detected objects.

## Demo

### Sample Outputs

![output_small](https://github.com/user-attachments/assets/d28132e6-75af-4d21-881f-f1d10210fc12)

![output_small2](https://github.com/user-attachments/assets/8a2d352d-55fd-4628-8956-e7cabf91979a)

![output_small3](https://github.com/user-attachments/assets/5d47dc13-912c-48c9-83b3-6bf5880f41e5)

## Requirements

- Python 3.x
- OpenCV
- NumPy

### Download YOLOv3 Weights
Download the pre-trained YOLOv3 weights from the following link:  
[Download YOLOv3 Weights](https://pjreddie.com/media/files/yolov3.weights)

## Usage

To run the object detection on a video stream, you can modify the `video_source` parameter. By default, it is set to `0`, which uses the webcam. If you want to process a specific video file, change `0` to the desired file path. 

Example:

```python
video_processor = VideoProcessor(video_source="object detection test.mp4", output_file="output.avi")

