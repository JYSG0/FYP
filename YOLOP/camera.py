import cv2
import os

# Open the default webcam (index 2)
cap = cv2.VideoCapture(6)

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

# Initialize variables
recording = False
out = None

# Loop to capture and control recording
while cap.isOpened():
    ret, frame = cap.read()  # Capture frame-by-frame
    if ret:
        # Display the frame
        cv2.imshow('Webcam', frame)

        # If recording, write the frame to the output file
        if recording:
            out.write(frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        
        # 'r' key to start/stop video recording
        if key == ord('r'):
            if not recording:
                print("Recording started...")
                vid_name = os.path.join(output_video_dir, f"output_video_{vid_counter}.mp4")  # Set video name and counter
                # Define the codec and create a VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(vid_name, fourcc, 20.0, (640, 480))
                recording = True
            else:
                print("Recording stopped.")
                recording = False
                print(f"{vid_name} written!")
                vid_counter += 1
                out.release()  # Stop recording and release the output file

        # 's' key to capture an image
        elif key == ord('s'):
            img_name = os.path.join(output_image_dir, f"opencv_frame_{img_counter}.png")  # Set image name and counter
            cv2.imwrite(img_name, frame)  # Write image file
            print(f"{img_name} written!")
            img_counter += 1

        # Break the loop if 'ESC' is pressed
        elif key == 27:  # ESC key
            print("Exiting...")
            break
        elif key == ord('q'):  # ord('q') gets the ASCII for 'q'
            print("'q' key pressed. Exiting...")
            break
    else:
        break

# Release the webcam and any output file, and close windows
cap.release()
if recording:
    out.release()
cv2.destroyAllWindows()
