import cv2
import numpy as np

# Load YOLOv4 pre-trained model
yolo_weights = 'models/yolov3.weights'
yolo_cfg = "models/yolov3.cfg"
yolo_net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

# Load class labels (COCO dataset)
with open("models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Open video file (replace with your video path or use webcam)
video_path = "videos/1903270-uhd_1920_1440_30fps.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to save the output video
output_video_path = "outputs/human_detection_output.avi"  # Save as .avi file
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Using XVID codec for AVI format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Prepare image for YOLOv4 (resize and normalize)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set the blob as input to the network
    yolo_net.setInput(blob)

    # Get output layers
    output_layers = yolo_net.getUnconnectedOutLayersNames()

    # Run detection
    layer_outputs = yolo_net.forward(output_layers)

    # Initialize lists for detected objects
    boxes = []
    confidences = []
    class_ids = []

    # Loop through detections
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Only consider human (class ID = 0 for 'person')
            if confidence > 0.5 and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Get coordinates for the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Store the box, confidence, and class ID
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to remove redundant boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Check if any detections are made
    if len(indices) > 0 and indices.size > 0:
        for i in indices.flatten():  # Flatten the indices to handle the list properly
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]  # Use the default label for detection (e.g., "person")

            # Draw bounding box and label (class name)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Write the processed frame to the output video
    out.write(frame)

# Release video capture and writer
cap.release()
out.release()

# # Close any OpenCV windows (if used locally)
cv2.destroyAllWindows()

print(f"Output video saved to: {output_video_path}")
