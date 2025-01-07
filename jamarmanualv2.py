# main.py
import socket
import odrive
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
import keyboard
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
import threading
import matplotlib.pyplot as plt
# from source import FastestRplidar
import pygame

# list_board_pins()
i2c=busio.I2C(board.SCL_1, board.SDA_1)
time.sleep(0.1)  # Small delay to stabilize the connection
ads=ADS.ADS1115(i2c, address=0x48)

chan1 = AnalogIn(ads, ADS.P0)
chan2=AnalogIn(ads, ADS.P3)
print(chan1.value, chan1.voltage)
print(chan2.value, chan2.voltage)



# Initialize pygame
pygame.init()

# Get screen resolution for fullscreen mode
info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h

# Load the model
model = YOLO("best.pt")
names = model.names
# global cap for camera
cap = DepthCamera()

cap1 = cv2.VideoCapture(6)


# Function to map chan1 value to a specific range (e.g., 10 to 100)
def map_value(value, input_min, input_max, output_min, output_max):
    return ((value - input_min) * (output_max - output_min)) / (input_max - input_min) + output_min

# Initialise GPIO Pins
pwm = digitalio.DigitalInOut(board.D22)
steering = digitalio.DigitalInOut(board.D13)
pwm.direction = digitalio.Direction.OUTPUT
steering.direction = digitalio.Direction.OUTPUT

# Initialise actuator IDLE 
pwm.value = False  # Set GPIO12 high
steering.value = False  # Set GPIO13 high

# Global variable to track drivable area detection and object detection
Steering = False
start_mode = True
speed_mode = False
steering_angle = 0
pot_value = 0
input_velocity = 1  # Global variable to track velocity for motors

#object detection class_id
class_ids=[]

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

# class Lidar:

#     def get_data(self):
#         """Fetch scan data from the LiDAR."""
#         return self.lidar.get_scan_as_xy(filter_quality=True)

#     def stop(self):
#         """Stop the LiDAR motor."""
#         self.lidar.stopmotor()

#     def __init__(self, max_retries=5, retry_delay=2):
#         """
#         Initialize the Lidar class with retry functionality.
        
#         :param max_retries: Maximum number of retry attempts.
#         :param retry_delay: Delay (in seconds) between retries.
#         """
#         self.lidar = FastestRplidar()
#         self.max_retries = max_retries
#         self.retry_delay = retry_delay

#         self.connect_with_retry()

#     def connect_with_retry(self):
#         """Attempt to connect to the LiDAR with retries."""
#         attempts = 0
#         while attempts < self.max_retries:
#             try:
#                 print(f"Connecting to LiDAR (Attempt {attempts + 1}/{self.max_retries})...")
#                 self.lidar.connectlidar()
#                 # Start the LiDAR motor
#                 self.lidar.startmotor(my_scanmode=2)
#                 # Verify the connection (example: fetch a sample scan)
#                 test_data = self.lidar.get_scan_as_xy(filter_quality=False)
#                 if test_data:  # If data is received, the connection is valid
#                     print("Connection successful!")
#                     return
#                 else:
#                     raise RuntimeError("Connection verification failed. No data received.")
            
                
#             except Exception as e:
#                 attempts += 1
#                 print(f"Connection failed: {e}. Retrying in {self.retry_delay} seconds...")
#                 time.sleep(self.retry_delay)

#         # If all attempts fail, raise an exception
#         raise RuntimeError("Failed to connect to LiDAR after multiple attempts.")


# def is_point_in_triangle(x, y):
#     """Check if a point (x, y) is inside the triangle."""
#     # Conditions for the triangle:
#     # 1. Below y = 1.2x (Red line)
#     # 2. Above y = -1.2x (Purple line)
#     # 3. Above y = -1250 (Horizontal black line)
#     return y <= 1.2 * x and y <= -1.2 * x and y >= -1250 and y < 0


# def detect_zones():
#     global lidar
#     lidar = Lidar()
#     # # Set up the matplotlib figure and axis
#     # fig, ax = plt.subplots()
#     # ax.set_xlim(6000, -6000)
#     # ax.set_ylim(6000, -6000)
#     # ax.set_xlabel("X-axis (mm)")
#     # ax.set_ylabel("Y-axis (mm)")
#     # ax.set_title("Real-Time LiDAR Scan")
#     # scatter = ax.scatter([], [], s=1)  # Initialize scatter plot

#     # # Plot line using an expression (e.g., y = 1.2x)
#     # x_line = np.linspace(-6000, 6000, 500)  # Generate X values from -6000 to 6000
#     # y_line = 1.2 * x_line  # Calculate Y values for the red line
#     # ax.plot(x_line, y_line, color='red', linewidth=1, label='y = 1.2x')  

#     # # Plot line using an expression (e.g., y = -1.2x)
#     # y_line1 = -1.2 * x_line  # Calculate Y values for the purple line
#     # ax.plot(x_line, y_line1, color='purple', linewidth=1, label='y = -1.2x ')  

#     # # Draw a horizontal line at y = -1250
#     # ax.axhline(y=-1250, color='black', linestyle='--', linewidth=1, label='y = -1250')  

#     # ax.legend()  # Add a legend to distinguish points and the line

#     try:
#         while True:
#             # Get LiDAR data
#             scan_data = lidar.get_data()
#             as_np = np.asarray(scan_data)
#             if as_np.size == 0 or len(as_np.shape) != 2 or as_np.shape[1] != 2:
#                 print("Invalid LiDAR data, skipping this cycle.")
#                 lidar.stop()
#                 break

#             # Update plot data
#             x_data = -as_np[:, 1]
#             y_data = -as_np[:, 0]
#             # scatter.set_offsets(np.c_[x_data, y_data])

#             # Check each point to see if it is inside the triangle
#             for x, y in zip(x_data, y_data):
#                 if is_point_in_triangle(x, y):
#                     print(f"Point inside triangle detected: ({x:.2f}, {y:.2f})")
#                     odrv0.axis1.controller.input_vel = 0
#                     odrv0.axis0.controller.input_vel = 0
                

#             # Update the plot
#             plt.pause(0.1)  # Add a small delay for smooth updating

#     except KeyboardInterrupt:
#         print("Exiting program...")
#     finally:
#         # Stop the LiDAR before exiting
#         lidar.stop()
#         # plt.close(fig)

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


#gets centre of bounding bvox
def get_center_of_bbox(box):
    """ Calculate the center point of the bounding box """
    x_center = int((box[0] + box[2]) / 2) 
    y_center = int((box[1] + box[3]) / 2) 
    return (x_center, y_center)

def camera(cap1):
    global frame1
    if True:
        ret1, frame1 = cap1.read()
        idx = 0

        if not ret1:
            print("Failed to grab frame")
            return

# Connect to ODrive and ESP
odrv0 = connect_to_odrive()
last_voltage_print_time = time.time()  # Initialize the timer

# Display drive voltage
print("ODrive Voltage:", odrv0.vbus_voltage)
odrv0.axis0.controller.config.control_mode = 2 # Vel Mode
odrv0.axis0.controller.config.input_mode = 2 #Vel Mode
odrv0.axis0.requested_state = 8  # Start Motor
odrv0.axis1.requested_state = 8 # Start Motor


def detection(cap, run_detection=True):
    #global variables to pass
    global class_ids, annotated_frame

    if True:
        ret, success, frame = cap.get_frame()
        idx = 0
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


            if boxes is not None:
                for box, cls in zip(boxes, clss):
                    idx += 1
                    annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                    crop_obj = annotated_frame[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
                    center_point = get_center_of_bbox(box)
                    # x_center = center_point[0] * (screen_width/640)
                    # y_center = center_point[1] * (screen_height/480)
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
        

        # Resize the frame to match the screen resolution
        resized_frame = cv2.resize(annotated_frame, (screen_width, screen_height))



        # Create the OpenCV window
        cv2.namedWindow('YOLOv8 Detection', cv2.WND_PROP_FULLSCREEN)


        # Set the window to fullscreen mode
        cv2.setWindowProperty('YOLOv8 Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Display the annotated frame
        cv2.imshow('YOLOv8 Detection', resized_frame)  

def Manual_drive(class_ids):
    global Steering, start_mode, pot_value, steering_angle, input_velocity,speed_mode

    # Read potentiometer value
    try:
        pot_value = map_value(chan1.value, 0, 26230, 0, 1023)
        steering_angle = map_value(pot_value, 0, 1023, -40, 40)
    except OSError as e:
        print(f"Error reading potentiometer: {e}")
        pot_value, steering_angle = None, None

    # Default behavior: Stop motors if no keys are pressed
    odrv0.axis1.controller.input_vel = 0
    odrv0.axis0.controller.input_vel = 0
    pwm.value = False
    steering.value = False
    # Handle specific key presses
    if keyboard.is_pressed('e'):
        start_mode = not start_mode

        if start_mode:  # Start motors
            print("Starting...")
            odrv0.axis0.requested_state = 8  # Start Motor
            odrv0.axis1.requested_state = 8  # Start Motor
        else:  # Stop motors
            print("Stopping...")
            odrv0.axis1.requested_state = 1  # Set ODrive to idle state
            odrv0.axis0.requested_state = 1  # Set ODrive to idle state

    if keyboard.is_pressed('w'):
        odrv0.axis1.controller.input_vel = -input_velocity
        odrv0.axis0.controller.input_vel = -input_velocity
        if input_velocity == 1:

            #Sharp Left Turn
            if keyboard.is_pressed('a'):
                odrv0.axis0.controller.input_vel = -1
                odrv0.axis1.controller.input_vel = -0.5
            #Sharp Right Turn
            if keyboard.is_pressed('d'):
                odrv0.axis1.controller.input_vel = -1
                odrv0.axis0.controller.input_vel = -0.5

        if input_velocity == 2:
            #Sharp Left Turn
            if keyboard.is_pressed('a'):
                odrv0.axis0.controller.input_vel = -2
                odrv0.axis1.controller.input_vel = -1
            #Sharp Right Turn
            if keyboard.is_pressed('d'):
                odrv0.axis1.controller.input_vel = -2
                odrv0.axis0.controller.input_vel = -1

        if odrv0.axis1.controller.input_vel >= -2:

            # People or Stop sign detected
            if 4 in class_ids or 8 in class_ids:
                print("Stop")
                odrv0.axis1.controller.input_vel = 0
                odrv0.axis0.controller.input_vel = 0

            # Slow or Speed sign or Hump or Pedestrain Sign detected
            elif 6 in class_ids or 7 in class_ids or 1 in class_ids or 3 in class_ids:
                print("Slow")
                odrv0.axis1.controller.input_vel = -0.8
                odrv0.axis0.controller.input_vel = -0.8

    if keyboard.is_pressed('s'):
        odrv0.axis1.controller.input_vel = input_velocity
        odrv0.axis0.controller.input_vel = input_velocity
        print(f"BACKWARD at speed: {input_velocity}")

    # 'shift' key to brake
    if keyboard.is_pressed('shift'):
        odrv0.axis1.controller.input_vel = 0
        odrv0.axis0.controller.input_vel = 0
        input_velocity = 1
        
        print ('STOP')
    if 2 in class_ids and steering_angle is not None and steering_angle <= 39:
        pwm.value = True
        steering.value = False
        print(f"Steering Left: Potentiometer Value: {int(steering_angle)}")

    if 5 in class_ids and steering_angle is not None and steering_angle >= -25:
        pwm.value = True
        steering.value = True
        print(f"Steering Left: Potentiometer Value: {int(steering_angle)}")

    if keyboard.is_pressed('a') and steering_angle is not None and steering_angle <= 39:  # Steer Left
        pwm.value = True
        steering.value = False
        print(f"Steering Left: Potentiometer Value: {int(steering_angle)}")

    if keyboard.is_pressed('d')  and steering_angle is not None and steering_angle >= -25:  # Steer Right
        pwm.value = True
        steering.value = True
        print(f"Steering Right: Potentiometer Value: {int(steering_angle)}")

    # Spacebar to increase input velocity
    if keyboard.is_pressed('space')  and not speed_mode:
        speed_mode = True
        if input_velocity == 1:
            input_velocity = 2
            print(f"Increased velocity to: {input_velocity}")
        else:
            input_velocity = 1
            print(f"Increased velocity to: {input_velocity}")

    # Reset `speed_mode` once the spacebar is released
    if not keyboard.is_pressed('space') :
        speed_mode = False

    if pot_value is None:
        return        
    if steering_angle is None:
        return
    return pot_value
    
def main():

    # global exit variable
    global exit_flag, recording, out_annotated, out_combined, img_counter, vid_counter,frame1,steering_angle, last_voltage_print_time, current_time
    # Start lane detection in a separate thread or process
    # Main event loop
    while not exit_flag:
        # Print the ODrive voltage at a 1-second interval
        current_time = time.time()
        if current_time - last_voltage_print_time >= 0.3:  # 0.3 second interval
            print("ODrive Voltage:", odrv0.vbus_voltage)
            last_voltage_print_time = current_time        # Poll for pygame events
        if steering_angle is None:
            return
        
        detection(cap)

        camera(cap1)

        Manual_drive(class_ids)
        # Check for 'q' key in pygame to exit
        if keyboard.is_pressed('q'):
            print("Resetting")
            odrv0.axis1.controller.input_vel = 0
            odrv0.axis0.controller.input_vel = 0
            if odrv0.axis0.controller.input_vel == 0 or odrv0.axis1.controller.input_vel == 0:
                odrv0.axis1.requested_state = 3  # Set ODrive to idle state
                odrv0.axis0.requested_state = 3  # Set ODrive to idle state

        pot_value = Manual_drive(class_ids)
        if pot_value == None:
            continue
        
        # If recording, write the frame to the video output file
        if recording:
            out_combined.write(frame1)
            out_annotated.write(annotated_frame)
        # Break the loop if 'ESC'
        elif keyboard.is_pressed('esc'):  # ESC to exit and restart motor
            print("Exiting...")
            odrv0.axis1.controller.input_vel = 0
            odrv0.axis1.requested_state = 1  # Set ODrive to idle state
            odrv0.axis0.controller.input_vel = 0
            odrv0.axis0.requested_state = 1  # Set ODrive to idle state
            exit_flag = True       
            break

        #Recording functionality with 'r' key for both frames
        if keyboard.is_pressed('r'): 
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
        if keyboard.is_pressed('t'): 
            img_name_annotated = os.path.join(output_image_dir, f"object_frame_{img_counter}.png")
            img_name_combined = os.path.join(output_image_dir, f"lane_frame_{img_counter}.png")
            cv2.imwrite(img_name_annotated, annotated_frame)  # Save YOLO-detected frame
            cv2.imwrite(img_name_combined, frame1)
            print(f"Images saved: {img_name_annotated} and {img_name_combined}")
            img_counter += 1
   
        # 'l' key to see status
        elif keyboard.is_pressed('l') : 
            print("Status:")
            print ("recording:",[recording])
            print(f"{odrv0.axis1.controller.input_vel}")
            print(f"{odrv0.axis0.controller.input_vel}")
        cv2.waitKey(1)
    #Quit all components
    cap.release()
    cap1.release()
    cv2.destroyAllWindows()  # Close windows after exiting
    # lidar.stop()
    print("Exiting all components.")


exit_flag = False

if  __name__ == '__main__':
    print("Starting concurrent LIDAR and Camera detection...")

    # Start YOLO detection thread
    detection_thread = threading.Thread(target=main, daemon=True)
    # Start LIDAR detection thread
    # lidar_thread = threading.Thread(target=detect_zones, daemon=True)

    # Start both threads
    detection_thread.start()
    # lidar_thread.start()

    # Wait for threads to finish (Ctrl+C to stop)
    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Stopping all threads...")
        # Stop the LiDAR before exiting
        # lidar.stop()
    # main()
