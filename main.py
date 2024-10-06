import cv2
import numpy as np
import json

class YOLODetector:
    def __init__(self, weights_path, config_path, names_path, gpu=False):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        if gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        with open(names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def process_frame(self, img):
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:

                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Perform non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        detections = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = self.colors[class_ids[i]]

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

                detections.append({
                    'class': label,
                    'confidence': confidence,
                    'box': (x, y, w, h)
                })
        
        return img, detections


class VideoProcessor:
    def __init__(self, video_source=0, output_file="output.avi"):
        self.cap = cv2.VideoCapture(video_source) 
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(output_file, self.fourcc, 20.0, (640, 480))

    def process(self, detector: YOLODetector):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))
            
            processed_frame, detected_objects = detector.process_frame(frame)
            cv2.imshow("Object Detection", processed_frame)
            
            self.out.write(processed_frame)
            # Save detections to JSON file
            with open('detections.json', 'w') as f:
                json.dump(detected_objects, f)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Initialize YOLO detector
    yolo_detector = YOLODetector(
        weights_path='yolov3.weights',
        config_path='yolov3.cfg',
        names_path='coco.names',
        gpu=True  # Set to False if GPU is not available
    )
    
    video_processor = VideoProcessor(video_source=0, output_file="output.avi")
    video_processor.process(yolo_detector)
