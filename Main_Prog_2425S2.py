import cv2
import os
#from ultralytics import YOLO
import odrive
import Jetson.GPIO as GPIO
import pygame
from ultralytics import YOLO

 

# Function to find and connect to ODrive
def connect_to_odrive():
    try:
        odrv = odrive.find_any()
        if odrv is None:
            print("ODrive not found. Exiting program.")
            exit()
        return odrv
    except Exception as e:
        print("Error connecting to ODrive:", e)
        exit()

def Detect(threshold, allowed_classes):
    global people_detected
    global chair_detected
    global people_count
    global chair_count
    people_count = 0
    chair_count = 0 
    # Access the detections
    annotated_frame = results[0].boxes  # Get bounding boxes

    # Check if any objects are detected
    if annotated_frame is not None and len(annotated_frame) > 0:
        for box in annotated_frame.data.tolist():  # Convert tensor to list            
            class_id = int(box[5])  # Index for class ID
            confidence = box[4]      # Index for confidence
            # Check if the detected class is in the allowed classes
            if (allowed_classes is None or class_id in allowed_classes) and confidence > threshold:
                if class_id == 0:  # Assuming 0 is the class ID for people
                    people_count += 1
                    people_detected = True

                if class_id == 56:  # Assuming 56 is the class ID for chairs
                    chair_count += 1
                    chair_detected = True
        # Print appropriate messages based on the counts
        if people_count > 0:
            print(f"Detected {people_count} people.")
        elif chair_count > 0:
            print(f"Detected {chair_count} chairs.")        
        return people_count, chair_count
    
    else:
        people_detected = False
        chair_detected = False

people_detected = False
chair_detected = False
# Connect to ODrive
odrv0 = connect_to_odrive()

# Initialize pygame
pygame.init()

#Initialise GPIO Pins
pwm = 12
steering = 13
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup([pwm, steering], GPIO.OUT, initial=GPIO.LOW)

# Define window size (To capture events)
screen = pygame.display.set_mode((640,480))
# Display drive voltage
print("ODrive Voltage:", odrv0.vbus_voltage)

# Open the default webcam (index 1)
cap = cv2.VideoCapture(0)

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

# Load the YOLOv8 model (YOLOv8s model used here; you can choose a different one)
model = YOLO('yolov8s.pt')

# Loop to capture, detect, and control recording
while cap.isOpened():
    ret, frame = cap.read()  # Capture frame-by-frame
    if ret:
                # Handle event queue
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        # YOLOv8 object detection
        threshold = 0.5 # conf threshold
        people = [0] #class id of people
        chair = [56] #class id of chair
            
        results = model(frame, conf=threshold, verbose=False) #Running detection
        # Annotate the frame with bounding boxes and labels
        annotated_frame = results[0].plot()
        Detect(threshold, people)
        Detect(threshold, chair)

        if people_detected == True:
            odrv0.axis1.controller.input_pos == 0
            odrv0.axis0.controller.input_pos == 0
        # Display the annotated frame
        cv2.imshow('YOLOv8 Detection', annotated_frame)

        # If recording, write the frame to the video output file
        if recording:
            out.write(annotated_frame)

        # 'r' key to start/stop video recording
        if keys[pygame.K_r]:
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

        # 't' key to capture an image
        elif keys[pygame.K_t]:
            img_name = os.path.join(output_image_dir, f"opencv_frame_{img_counter}.png")  # Set image name
            cv2.imwrite(img_name, annotated_frame)  # Write image file
            print(f"{img_name} written!")
            img_counter += 1

        # 'e' key to start motor
        elif keys[pygame.K_e]:  
            print("Starting...")
            odrv0.axis0.requested_state = 8  # Reactivate ODrive
            odrv0.axis1.requested_state = 8 # Reactivate ODrive

        # 'q' is pressed
        elif keys[pygame.K_q]:  # 'q' key to restart motor
            print("Resetting...")
            odrv0.axis1.controller.input_pos = 0
            odrv0.axis1.requested_state = 3  # Set ODrive to idle state
            odrv0.axis0.controller.input_pos = 0
            odrv0.axis0.requested_state = 3  # Set ODrive to idle state
        # Break the loop if 'ESC'
        elif keys[pygame.K_ESCAPE]:  # ESC to exit and restart motor
            print("Exiting...")
            odrv0.axis1.controller.input_pos = 0
            odrv0.axis1.requested_state = 3  # Set ODrive to idle state
            odrv0.axis0.controller.input_pos = 0
            odrv0.axis0.requested_state = 3  # Set ODrive to idle state
            break

        # 'w' key to move forward
        if keys[pygame.K_w]:
            odrv0.axis1.controller.input_pos += 0.1
            odrv0.axis0.controller.input_pos -= 0.1
            print ('FORWARD')
        # 's' key to move backwards
        if keys[pygame.K_s]:
            odrv0.axis1.controller.input_pos -= 0.1
            odrv0.axis0.controller.input_pos += 0.1
            print ('BACKWARD')
        #'a' is pressed
        if keys[pygame.K_a]:  # Steer Left
            GPIO.output(pwm, GPIO.LOW)
            GPIO.output(steering, GPIO.HIGH)
            print ('LEFT')

        #'d' is pressed
        if keys[pygame.K_d]:  # Steer Right
            GPIO.output(pwm, GPIO.LOW)
            GPIO.output(steering, GPIO.LOW) 
            print('RIGHT')
        if not keys[pygame.K_a] and not keys[pygame.K_d]:  # Joystick IDLE
            GPIO.output(pwm, GPIO.HIGH)
            GPIO.output(steering, GPIO.HIGH)
    else:
        break

# Release the webcam and any output file, and close windows
cap.release()
pygame.quit()
if recording:
    out.release()
cv2.destroyAllWindows()

