import cv2
import os
from ultralytics import YOLO
import odrive
import Jetson.GPIO as GPIO
import pygame
from ultralytics import YOLO
from collections import Counter
 

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

#Connect to ODrive
odrv0 = connect_to_odrive()


def Manual_drive(keys):
         # 'e' key to start motor
        if keys[pygame.K_e]:  
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
            Steering = True
        #'d' is pressed
        if keys[pygame.K_d]:  # Steer Right
            GPIO.output(pwm, GPIO.LOW)
            GPIO.output(steering, GPIO.LOW) 
            print('RIGHT')
            Steering = True
        if not keys[pygame.K_a] and not keys[pygame.K_d]:  # Joystick IDLE
            GPIO.output(pwm, GPIO.HIGH)
            GPIO.output(steering, GPIO.HIGH)
            Steering = False


def Lane():
    lane =+ 1
def Auto():
    auto =+ 1
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

# Initialize a flag for Lane assist, autot driving , run _detection
run_detection = True
Auto_driving = True
Steering = False
# Load the YOLOv8 model (YOLOv8s model used here; you can choose a different one)
model = YOLO('best.pt')

# Loop to capture, detect, and control recording
while cap.isOpened():
    ret, frame = cap.read()  # Capture frame-by-frame
    if ret:
        # Handle event queue
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        if run_detection:
            results = model(frame, conf=0.5, verbose=False, device=0) #Running detection
            annotated_frame = results[0].plot() # Annotate the frame with bounding boxes and labels
                                # Print class names for each detection
            for result in results:
                class_ids = result.boxes.cls.tolist()
                class_names = [model.names[int(id)] for id in class_ids]  # Map IDs to names

    
            # Use Counter to count occurrences of each class name
            class_counts = Counter(class_names)
            # Print the count for each class detected
            for class_name, count in class_counts.items():
                print(f"{count} {class_name} detected.")
        else:
            results = model(frame, conf=0.5, verbose=False, device=0) #Running detection
            annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow('YOLOv8 Detection', annotated_frame)


        
        if Auto_driving and not keys[pygame.K_m]:
            Auto()
            Lane()
            if 4 in class_ids or 8 in class_ids:            
                print ("Stop")
                Auto_driving = False
                run_detection = False
                continue
            
        elif not Auto_driving and not keys[pygame.K_m]:
            Manual_drive(keys)
            

        elif keys[pygame.K_m]:
            print("Override")
            Auto_driving = True
            run_detection = True

        # If recording, write the frame to the video output file
        if recording:
            out.write(annotated_frame)

        # Break the loop if 'ESC'
        elif keys[pygame.K_ESCAPE]:  # ESC to exit and restart motor
            print("Exiting...")
            odrv0.axis1.controller.input_pos = 0
            odrv0.axis1.requested_state = 3  # Set ODrive to idle state
            odrv0.axis0.controller.input_pos = 0
            odrv0.axis0.requested_state = 3  # Set ODrive to idle state
            break
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

        # Break the loop if 'ESC'
        elif keys[pygame.K_l]:  # ESC to exit and restart motor
            print("Status:")
            print ("Auto:",[{Auto_driving}])
            print ("Detection:",[{run_detection}])
            print ("recording:",[{recording}])

        # 't' key to capture an image
        elif keys[pygame.K_t]:
            img_name = os.path.join(output_image_dir, f"opencv_frame_{img_counter}.png")  # Set image name
            cv2.imwrite(img_name, annotated_frame)  # Write image file
            print(f"{img_name} written!")
            img_counter += 1
        
        cv2.waitKey(1)

 
# Release the webcam and any output file, and close windows
cap.release()
pygame.quit()
if recording:
    out.release()
cv2.destroyAllWindows()

