import cv2
from ultralytics import YOLO
import os

# Load the YOLOv8 model
model = YOLO("runs/detect/train8/weights/best.pt")

# Create directories for saving images and coordinates if they don't exist
os.makedirs("saved_images", exist_ok=True)
os.makedirs("saved_coordinates", exist_ok=True)

# Open the video file or webcam (0 for webcam)
cap = cv2.VideoCapture(0)

# Initialize a counter for saved images and coordinates
save_counter = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Check if a key is pressed
        key = cv2.waitKey(1) & 0xFF

        # Save the image and coordinates if 's' is pressed
        if key == ord('s'):
            # Increment the save counter
            save_counter += 1

            # Save the current frame as an image file
            image_filename = f"saved_images/frame_{save_counter}.jpg"
            cv2.imwrite(image_filename, frame)

            # Open a text file to save coordinates
            coordinate_filename = f"saved_coordinates/coordinates_{save_counter}.txt"
            with open(coordinate_filename, "w") as file:
                for result in results:
                    boxes = result.boxes.xyxy  # Get bounding box coordinates (x1, y1, x2, y2)
                    classes = result.boxes.cls
                    for box in boxes:
                        # Write coordinates to file
                        file.write(f"class:{classes} x1: {box[0].item()}, y1: {box[1].item()}, x2: {box[2].item()}, y2: {box[3].item()}\n")
            
            print(f"Saved frame and coordinates as {image_filename} and {coordinate_filename}.")

        # Break the loop if 'q' is pressed
        elif key == ord('q'):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
