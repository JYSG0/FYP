import cv2
import os
from ultralytics import YOLO

# Open the default webcam (index 2)
cap = cv2.VideoCapture(2)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create directories for output if they don't exist
output_image_dir = 'output_images'
output_video_dir = 'output_videos'
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_video_dir, exist_ok=True)

# Get the highest existing image counter in the output directory
existing_images = [f for f in os.listdir(output_image_dir) if f.startswith("opencv_frame_") and f.endswith(".png")]
if existing_images:
    img_counter = max([int(f.split('_')[2].split('.')[0]) for f in existing_images]) + 1
else:
    img_counter = 0

# Get the highest existing video counter in the output directory
existing_videos = [f for f in os.listdir(output_video_dir) if f.startswith("output_video_") and f.endswith(".mp4")]
if existing_videos:
    vid_counter = max([int(f.split('_')[2].split('.')[0]) for f in existing_videos]) + 1
else:
    vid_counter = 0

# Initialize variables for recording
recording = False
out = None

# Load the YOLOv8 model (YOLOv8n model used here; you can choose a different one)
model = YOLO('best.pt')

# Specify the classes to filter (e.g., detecting only persons and cars)
# Set desired_classes to an empty list to detect all objects

# Loop to capture, detect, and control recording
while cap.isOpened():
    ret, frame = cap.read()  # Capture frame-by-frame
    if ret:
        # YOLOv8 object detection
        results = model(frame, conf = 0.5)
        # Annotate the frame with bounding boxes and labels
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow('YOLOv8 Detection', annotated_frame)

        # If recording, write the frame to the video output file
        if recording:
            out.write(annotated_frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF

        # 'r' key to start/stop video recording
        if key == ord('r'):
            if not recording:
                print("Recording started...")
                vid_name = os.path.join(output_video_dir, f"output_video_{vid_counter}.mp4")  # Set video name
                # Define the codec and create a VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(vid_name, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                recording = True
            else:
                print("Recording stopped.")
                recording = False
                print(f"{vid_name} written!")
                vid_counter += 1
                out.release()  # Stop recording and release the output file

        # 's' key to capture an image
        elif key == ord('s'):
            img_name = os.path.join(output_image_dir, f"opencv_frame_{img_counter}.png")  # Set image name
            cv2.imwrite(img_name, annotated_frame)  # Write image file
            print(f"{img_name} written!")
            img_counter += 1

        # Break the loop if 'ESC' or 'q' is pressed
        elif key == 27 or key == ord('q'):  # ESC or 'q' key to exit
            print("Exiting...")
            break
    else:
        break

# Release the webcam and any output file, and close windows
cap.release()
if recording:
    out.release()
cv2.destroyAllWindows()
