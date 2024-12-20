# main.py
import socket
import odrive
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
import pygame
import time
import os
import pyrealsense2
from realsense_depth import *
from ultralytics.utils.plotting import Annotator, colors
import Jetson.GPIO as GPIO
import busio
import board
import digitalio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# list_board_pins()

i2c=busio.I2C(board.SCL_1, board.SDA_1)
time.sleep(0.1)  # Small delay to stabilize the connection
ads=ADS.ADS1115(i2c, address=0x48)

chan1 = AnalogIn(ads, ADS.P0)
chan2=AnalogIn(ads, ADS.P3)
print(chan1.value, chan1.voltage)
print(chan2.value, chan2.voltage)

#  Function to map chan1 value to a specific range (e.g., 10 to 100)
def map_value(value, input_min, input_max, output_min, output_max):
    return ((value - input_min) * (output_max - output_min)) / (input_max - input_min) + output_min

# Global variable to track drivable area detection and object detection
run_detection = True
Steering = False
start_mode = True

#object detection class_id
class_ids=[]

#Initialise GPIO Pins
pwm = digitalio.DigitalInOut(board.D12)
steering = digitalio.DigitalInOut(board.D13)
pwm.direction = digitalio.Direction.OUTPUT
steering.direction = digitalio.Direction.OUTPUT

#Initial Low 
pwm.value = True  # Set GPIO12 high
steering.value = True  # Set GPIO13 low

# global cap for camera
cap = DepthCamera()

cap1 = cv2.VideoCapture(6)

# Load the model
model = YOLO("best.pt")
names = model.names

# Directory setup for recording images and videos
output_image_dir = 'output_images'
output_video_dir = 'output_videos'
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_video_dir, exist_ok=True)

# Get the highest existing image counter in the output directory
existing_images = [f for f in os.listdir(output_image_dir) if f.startswith("object_frame_") and f.endswith(".png")]
if existing_images:
    img_counter = max([int(f.split('_')[2].split('.')[0]) for f in existing_images]) + 1
else:
    img_counter = 0

# Get the highest existing video counter in the output directory
existing_videos = [f for f in os.listdir(output_video_dir) if f.startswith("object_video_") and f.endswith(".mp4")]
if existing_videos:
    vid_counter = max([int(f.split('_')[2].split('.')[0]) for f in existing_videos]) + 1
else:
    vid_counter = 0

# Initialize counters and flags for recording and image capture
recording = False
out_annotated = None

# Initialize pygame
pygame.init()

# Get screen resolution for fullscreen mode
info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h

# Configure Pygame for a transparent window
os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
os.environ['SDL_VIDEO_CENTERED'] = "1"
pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN | pygame.SRCALPHA)

# Create a transparent surface for the overlay
overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)


# Lock mouse and keyboard to Pygame window
pygame.event.set_grab(True)


#gets centre of bounding bvox
def get_center_of_bbox(box):
    """ Calculate the center point of the bounding box """
    x_center = int((box[0] + box[2]) / 2) 
    y_center = int((box[1] + box[3]) / 2) 
    return (x_center, y_center)


def detection(cap, run_detection=True):
    #global variables to pass
    global class_ids, annotated_frame

    if True:
        ret, success, frame = cap.get_frame()
        idx = 0
        print(frame.shape)
        if not ret:
            print("Failed to grab frame")
            return


        # Run YOLO detection if enabled
        if run_detection:
            results = model(frame, conf=0.5, verbose=False, device=0)  # Running detection
            annotated_frame = results[0].plot()  # Annotate frame with segmentaion mask
            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(annotated_frame, line_width=2, example=names)
            annotated_frame = cv2.resize(annotated_frame, (screen_width, screen_height))            

            # Convert frame from BGR to RGB (OpenCV uses BGR by default, Pygame uses RGB)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Create a Pygame surface from the camera frame
            camera_surface = pygame.image.frombuffer(annotated_frame.tobytes(), (annotated_frame.shape[1], annotated_frame.shape[0]), "RGB")

            # Display the camera feed
            screen = pygame.display.get_surface()
            screen.blit(camera_surface, (0, 0))

            # Add transparent overlay (draw shapes, text, etc. on the overlay surface)
            overlay.fill((0, 0, 0, 0))  # Reset transparent surface
            screen.blit(overlay, (0, 0))  # Overlay on top of the camera feed

            # Update display
            pygame.display.update()

            if boxes is not None:
                for box, cls in zip(boxes, clss):
                    idx += 1
                    annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                    crop_obj = annotated_frame[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
                    center_point = get_center_of_bbox(box)
                    # x_center = center_point[0] * (screen_width/640)
                    # y_center = center_point[1] * (screen_height/480)
                    print(center_point)
                    cv2.circle(annotated_frame, center_point, 5, (0, 0, 255), -1)
                    distance = success[center_point[1] , center_point[0]]
                    cv2.putText(annotated_frame, "{}mm".format(distance), (center_point[0], center_point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            
            # Process and print class names for each detection
            for result in results:
                class_ids = result.boxes.cls.tolist()
                class_names = [model.names[int(id)] for id in class_ids]  # Map IDs to names
            
            # Count occurrences of each class name
            class_counts = Counter(class_names)
            for class_name, count in class_counts.items():
                print(f"{count} {class_name} detected.")


        else:
            results = model(frame, conf=0.5, verbose=False, device=0)  # Running detection
            annotated_frame = results[0].plot()


def camera(cap1):
    global frame1
    if cap1.isOpened():
        ret, frame1 = cap1.read()
        idx = 0

        if not ret:
            print("Failed to grab frame")
            return

        # # Display the annotated frame
        # cv2.imshow('Video', frame1)

# Connect to ODrive
def connect_to_odrive():
    print("Finding odrive")
    try:
        odrv = odrive.find_any()
        if odrv is None:
            print("ODrive not found. Exiting program.")
            exit()
        print("ODrive connected successfully.")
        return odrv
    except Exception as e:
        print("Error connecting to ODrive:", e)
        exit()


# Connect to ODrive 
odrv0 = connect_to_odrive()

# Display drive voltage
print("ODrive Voltage:", odrv0.vbus_voltage)
odrv0.axis0.controller.config.control_mode = 2 # Vel Mode
odrv0.axis0.controller.config.input_mode = 2 #Vel Mode
odrv0.axis0.requested_state = 8  # Start Motor
odrv0.axis1.requested_state = 8 # Start Motor


def Manual_drive(keys, class_ids):
        global Steering, start_mode

        # Read potentiometer value
        try:
            pot_value = map_value (chan1.value, 0, 26230, 0, 1023)
            steering_angle = map_value(pot_value, 0 , 1023, -40 ,40)
        except OSError as e:
            print(f"Error reading potentiometer: {e}")
    
        # 'e' key to start motor
        if keys[pygame.K_e]:  
            start_mode = not start_mode
            #Start motor
            if start_mode:
                print("Starting...")
                odrv0.axis0.requested_state = 8  # Start Motor
                odrv0.axis1.requested_state = 8 # Start Motor
                odrv0.axis1.controller.input_vel = 0
                odrv0.axis0.controller.input_vel = 0

            #Motor Idle
            if start_mode is False: 
                print("Stopping...")
                odrv0.axis1.controller.input_vel = 0
                odrv0.axis0.controller.input_vel = 0
                odrv0.axis1.requested_state = 1  # Set ODrive to idle state
                odrv0.axis0.requested_state = 1  # Set ODrive to idle state

        # 'w' key to move forward
        if keys[pygame.K_w]:
            odrv0.axis1.controller.input_vel = -1
            odrv0.axis0.controller.input_vel = -1  
            print ('FORWARD')

        if odrv0.axis1.controller.input_vel >= -1:

            # People or Stop sign detected
            if 4 in class_ids or 8 in class_ids:
                print("Stop")
                odrv0.axis1.controller.input_vel = 0
                odrv0.axis0.controller.input_vel = 0

            # Slow or Speed sign detected
            elif 6 in class_ids or 7 in class_ids:
                print("Slow")
                odrv0.axis1.controller.input_vel = -0.8
                odrv0.axis0.controller.input_vel = -0.8

            # Hump or Pedestrain Sign detected
            if 1 in class_ids or 3 in class_ids:
                print("Pedestrain")
                odrv0.axis1.controller.input_vel = -0.8
                odrv0.axis0.controller.input_vel = -0.8

        # 's' key to move backwards
        if keys[pygame.K_s]:
            odrv0.axis1.controller.input_vel = 1
            odrv0.axis0.controller.input_vel = 1      
            print ('BACKWARD')

        # 'shift' key to brake
        mods = pygame.key.get_mods()  # Get current modifier key states
        if mods & pygame.KMOD_SHIFT:
            odrv0.axis1.controller.input_vel = 0
            odrv0.axis0.controller.input_vel = 0
            print ('STOP')

             # 'a' is pressed
        if keys[pygame.K_a] and steering_angle <= 25:  # Steer Left
            pwm.value = False
            steering.value = True
            Steering = True
            if pot_value is not None:
                print(f"Steering Left: Potentiometer Value: {int(steering_angle)}")
            

        #'d' is pressed
        if keys[pygame.K_d] and steering_angle >= -25:  # Steer Right
            pwm.value = False
            steering.value = False
            Steering = True

            if pot_value is not None:
                print(f"Steering Right: Potentiometer Value: {int(steering_angle)}")

        if not keys[pygame.K_a] and not keys[pygame.K_d]: # Joystick IDLE
            pwm.value = True
            steering.value = True
            Steering = False

            if pot_value is not None:
                print(f"No Steering: Potentiometer Value: {int(steering_angle)}")
                
        if steering_angle >= 25 and keys[pygame.K_a]:  #Steer Right Limit
            pwm.value = True
            steering.value = True
            if pot_value is not None:
                print(f"No Steering: Potentiometer Value: {int(steering_angle)}")

        if steering_angle <= -25 and keys[pygame.K_d]: #Steer Left Limit
            pwm.value = True
            steering.value = True
            if pot_value is not None:
                print(f"No Steering: Potentiometer Value: {int(steering_angle)}")


        return pot_value

        
def main():
    # global exit variable
    global exit_flag, run_detection, recording, out_annotated, img_counter, vid_counter
    # Start lane detection in a separate thread or process
    # Main event loop
    while not exit_flag:
        # Poll for pygame events
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        detection(cap)
        # #Threading for detection and server_program
        # import threading
        # object_detection_thread = threading.Thread(target=detection(cap))
        # object_detection_thread.start()

        camera(cap1)

        Manual_drive(keys, class_ids)

        pot_value = Manual_drive(keys, class_ids)

        if pot_value == None:
            continue
        # If recording, write the frame to the video output file
        if recording:
            out_annotated.write(annotated_frame)
            out_combined.write(frame1)
        # Change Running Detection or not
        if keys[pygame.K_m]:
            run_detection = not run_detection # Toggle run_detection
            print (f"Object Detection is now: [{run_detection}]")
            odrv0.axis1.controller.input_vel = 0
            odrv0.axis1.controller.input_vel = 0

        # Break the loop if 'ESC'
        elif keys[pygame.K_ESCAPE]:  # ESC to exit and restart motor
            print("Exiting...")
            odrv0.axis1.controller.input_vel = 0
            odrv0.axis0.controller.input_vel = 0
            odrv0.reboot()
            exit_flag = True       
            break

        #Recording functionality with 'r' key for both frames
        if keys[pygame.K_r]:
            if not recording:
                print("Recording started for both annotated and combined frames...")
                # Set up video writers for both annotated and combined frames
                vid_name_annotated = os.path.join(output_video_dir, f"object_video_{vid_counter}.mp4")
                vid_name_combined = os.path.join(output_video_dir, f"lane_video_{vid_counter}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_combined = cv2.VideoWriter(vid_name_combined, fourcc, 20.0, 
                                                (frame1.shape[1], frame1.shape[0]))
                out_annotated = cv2.VideoWriter(vid_name_annotated, fourcc, 20.0, 
                                                (annotated_frame.shape[1], annotated_frame.shape[0]))
                recording = True
            else:
                print("Recording stopped.")
                recording = False
                out_annotated.release()  # Stop recording and release the output file for annotated frames
                out_combined.release()
                print(f"{vid_name_annotated} and {vid_name_combined} saved!")
                vid_counter += 1

        # Image capture functionality with 't' key for both frames
        elif keys[pygame.K_t]:
            img_name_annotated = os.path.join(output_image_dir, f"object_frame_{img_counter}.png")
            img_name_combined = os.path.join(output_image_dir, f"lane_frame_{img_counter}.png")
            cv2.imwrite(img_name_annotated, annotated_frame)  # Save YOLO-detected frame
            cv2.imwrite(img_name_combined, frame1)
            print(f"Images saved: {img_name_annotated} and {img_name_combined}")
            img_counter += 1

        # 'l' key to see status
        elif keys[pygame.K_l]: 
            print("Status:")
            print ("Detection:",[{run_detection}])
            print ("recording:",[{recording}])

        cv2.waitKey(1)
    #Quit all components
    pygame.quit()
    # cap.release()
    cap.release()
    cv2.destroyAllWindows()  # Close windows after exiting
    print("Exiting all components.")

exit_flag = False

if  __name__ == '__main__':

    main()