!pip install ultralytics 
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load YOLOv8 model (automatically downloads weights if not available)
model = YOLO('yolov8s.pt')  # Use 'yolov8n.pt', 'yolov8s.pt', etc., depending on your needs

# Load an image
# image_path = "cars.jpg"  # Replace with your image path
image_path = "persons.jpg"
results = model.predict(source=image_path, conf=0.4)  # Perform detection with confidence threshold 0.5

# Read the image using OpenCV for custom drawing
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for proper visualization

# Initialize human count
human_count = 0

# Iterate through detected objects
for result in results[0].boxes:
    label = int(result.cls)  # Class ID for the detected object
    if model.names[label] == 'human':  # Check if the detected object is a person
        human_count += 1

        # Get bounding box coordinates and confidence
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
        conf = result.conf[0]  # Confidence score

        # Draw a lighter border around the detected object
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Add smaller text for the label
        label_text = f"Person {conf:.2f}"  # Label with confidence
        font_scale = 0.5  # Smaller font size
        font_thickness = 2  # Lighter text
        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_w, text_h = text_size

        # Draw background for text
        cv2.rectangle(image, (x1, y1 - text_h - 5), (x1 + text_w, y1), (0, 255, 0), -1)  # Filled rectangle
        cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

# Print the count of humans detected
print(f"Number of Human detected: {human_count}")

# Visualize the results on the image
plt.imshow(image)
plt.axis('off')
plt.gca().spines['top'].set_visible(False)  # Remove top border
plt.gca().spines['right'].set_visible(False)  # Remove right border
plt.gca().spines['left'].set_visible(False)  # Remove left border
plt.gca().spines['bottom'].set_visible(False)  # Remove bottom border
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding around the image
plt.show()
