import cv2
import os

# Function to check video resolution and FPS
def get_video_info(capture):
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    return width, height, fps

# Open the video file (or replace with a webcam index, e.g., cap = cv2.VideoCapture(0))
cap = cv2.VideoCapture(2)

# Check if the video file is opened correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video resolution and FPS
width, height, fps = get_video_info(cap)
print(f"Video resolution: {width}x{height}, FPS: {fps}")

# Create directories for output if they don't exist
output_image_dir = 'output_images'
os.makedirs(output_image_dir, exist_ok=True)

# Get the highest existing image counter in the output directory
existing_images = [f for f in os.listdir(output_image_dir) if f.startswith("opencv_frame_") and f.endswith(".png")]
if existing_images:
    img_counter = max([int(f.split('_')[2].split('.')[0]) for f in existing_images]) + 1
else:
    img_counter = 0

# Calculate the delay between frames (in milliseconds) based on the video's FPS
frame_delay = int(1000 / fps)

# Initialize variables for recording
recording = False
frame_counter = 0

# Loop to capture and control recording
while cap.isOpened():
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break  # If no frame is captured, exit the loop

    # Check for key press
    key = cv2.waitKey(frame_delay) & 0xFF  # Adjust delay based on video FPS

    # 'r' key to start/stop recording
    if key == ord('r'):
        recording = not recording  # Toggle recording state
        if recording:
            print("Recording started...")
        else:
            print("Recording stopped.")

    # If recording, save each frame as an image (screenshot)
    if recording:
        # Only save one frame per second (every fps frames)
        if frame_counter % int(fps) == 0:

            img_name = os.path.join(output_image_dir, f"opencv_frame_{img_counter}.png")  # Set image name
            cv2.imwrite(img_name, frame)  # Save frame as image
            print(f"{img_name} written!")
            img_counter += 1

        # Increment the frame counter
        frame_counter += 1
    # Display the frame
    cv2.imshow('Video Frame', frame)

    # Break the loop if 'ESC' or 'q' is pressed
    if key == 27 or key == ord('q'):  # ESC or 'q' key to exit
        print("Exiting...")
        break

# Release the video file and close windows
cap.release()
cv2.destroyAllWindows()
