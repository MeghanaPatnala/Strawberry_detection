import cv2
from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("best.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Confidence threshold for detection
CONFIDENCE_THRESHOLD = 0.5

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model.predict(frame)

    # Process results
    for result in results:
        for box in result.boxes:
            confidence = box.conf.item()
            label = result.names[int(box.cls)]  # Get class label

            # Check confidence threshold
            if confidence > CONFIDENCE_THRESHOLD:
                # Extract coordinates and convert to integers
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                x_center = (x_min + x_max) // 2
                y_center = (y_min + y_max) // 2

                # Determine bounding box color based on ripeness
                if label.lower() == "ripe":
                    color = (0, 255, 0)  # Green for ripe
                    ripeness_text = "Ripe Strawberry"
                else:
                    color = (0, 0, 255)  # Red for unripe
                    ripeness_text = "Unripe Strawberry"

                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                # Display label, confidence, and coordinates
                text = f"{label}: {confidence:.2f}"
                coord_text = f"X: {x_center}, Y: {y_center}"

                cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, ripeness_text, (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, coord_text, (x_min, y_max + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Strawberry Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
