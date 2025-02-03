import sys
import odrive
import asyncio
import json
from fastapi import FastAPI, WebSocket
from uvicorn import Config, Server
import threading
import time
import keyboard

import Jetson.GPIO as GPIO
import busio
import board
import digitalio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
import os
from realsense_depth import *
from ultralytics.utils.plotting import Annotator, colors
import matplotlib.pyplot as plt
import pygame

# Globals
auto = False  # Manual mode by default
input_velocity = 1  # Default velocity
routeActive = False
connected_clients = []  # Store WebSocket clients
dir, angleToTurn, azimuth, turn = None, None, None, None  # Data from ESP32
speedMode = False
input_velocity = 1
within_tolerance = False
brozimuth = None
movement = None
peopleDetect = True

start_mode = True
steering_angle = None
steering_angle = 0
pot_value = 0

#Board pins
i2c=busio.I2C(board.SCL_1, board.SDA_1)
time.sleep(0.1)  # Small delay to stabilize the connection
ads=ADS.ADS1115(i2c, address=0x48)

chan1 = AnalogIn(ads, ADS.P0)
chan2=AnalogIn(ads, ADS.P3)
print(chan1.value, chan1.voltage)       
print(chan2.value, chan2.voltage)

#Initialise GPIO Pins
pwm = digitalio.DigitalInOut(board.D22)
steering = digitalio.DigitalInOut(board.D13)
pwm.direction = digitalio.Direction.OUTPUT
steering.direction = digitalio.Direction.OUTPUT

#Initial actuator IDLE 
pwm.value = False  # Set GPIO12 HIGH
steering.value = False  # Set GPIO13 HIGH

classIDs = []

# Initialize pygame
pygame.init()

# Get screen resolution for fullscreen mode
info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h

# FastAPI app
app = FastAPI()

#Load the AI model
model = YOLO("best.pt")
names = model.names
# global cap for camera
cap = DepthCamera()

cap1 = cv2.VideoCapture(6)

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

# Function to connect to ODrive
def connect_to_odrive():
    try:
        odrv = odrive.find_any()
        if odrv is None:
            print("ODrive not found. Exiting.")
            sys.exit()
        return odrv
    except Exception as e:
        print(f"Error connecting to ODrive: {e}")
        sys.exit()

# Initialize ODrive
odrv0 = connect_to_odrive()
odrv0.axis1.requested_state = 8
odrv0.axis0.requested_state = 8

last_voltage_print_time = time.time()  # Initialize the timer

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

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global dir, angleToTurn, turn, routeActive, minBrozimuth, maxBrozimuth, within_tolerance, brozimuth
    await websocket.accept()
    connected_clients.append(websocket)
    print("WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                print(message)
                message = json.loads(message)
                if message.get("type") == "vehicleControl":
                    routeActive = True
                    dir = message.get("jetson")
                    turn = message.get("modifier")
                    angleToTurn = message.get("angleToTurn")
                    within_tolerance = message.get("within_tolerance")

                    brozimuth = map_value(angleToTurn, -180, 180, -40, 40)
                    minBrozimuth = brozimuth - 3.5
                    maxBrozimuth = brozimuth + 3.5

                    print("brozimuth: ", brozimuth) # Angle to Turn but scaled to vehicle steering angle
                    print("Direction to turn to: ", dir) #direction to move
                    print("turn: ", turn)
                    print(f"Turn {angleToTurn} degrees") # Required Angle to turn for whole vehicle

            except json.JSONDecodeError:
                print("Invalid JSON received")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        connected_clients.remove(websocket)
        print("WebSocket client disconnected")

def map_value(value, input_min, input_max, output_min, output_max):
    return ((value - input_min) * (output_max - output_min)) / (input_max - input_min) + output_min


def detection(cap, run_detection=True):
    #global variables to pass
    global classIDs, annotated_frame

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
                    # Extract the region of the depth map corresponding to the bounding box

                    cv2.circle(annotated_frame, center_point, 5, (0, 0, 255), -1)

                    distance = success[center_point[1] , center_point[0]]
                    cv2.putText(annotated_frame, "{}mm".format(distance), (center_point[0], center_point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
      

            # Process and print class names for each detection
            for result in results:
                classIDs = result.boxes.cls.tolist()
                class_names = [model.names[int(id)] for id in classIDs]  # Map IDs to names
            
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


# Manual control
def manual_control():
    global input_velocity, start_mode, movement
    movement = None
    print("Manual mode: vehicle stopped")
    odrv0.axis1.controller.input_vel = 0
    odrv0.axis0.controller.input_vel = 0
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
        print("w pressed")
        movement = "forward"

        if keyboard.is_pressed('a'):
            movement = "left"

        elif keyboard.is_pressed('d'):
            movement = "right"

    if keyboard.is_pressed('s'):
        movement = "backward"

    print(movement)

    if movement == None:
        odrv0.axis1.controller.input_vel = 0
        odrv0.axis0.controller.input_vel = 0
    else:
        wheelsMovement(movement)
        steer()
    
    if keyboard.is_pressed('shift'):
        odrv0.axis1.controller.input_vel = 0
        odrv0.axis0.controller.input_vel = 0
        input_velocity = 1
        
        print ('STOP')

# Auto control
def auto_control():
    global dir, turn, input_velocity, movement
    print("Auto mode: Processing ESP32 data")

    #Axis 1 = right wheel
    #Axis 0 = left wheel

    time.sleep(1)

    if routeActive:
        if dir == "straight":
            movement = "forward"

        elif dir == "left":
            print("Left")
            movement = "left"

        elif dir == "right":
            print("right")
            movement = "right"

        wheelsMovement(movement)
        steer()
    else:
        odrv0.axis1.controller.input_vel = 0
        odrv0.axis0.controller.input_vel = 0

def wheelsMovement(movement):
    # print("Movement: ", movement)
    global classIDs, peopleDetect

    if 0 in classIDs:
        print("Go")
        odrv0.axis1.controller.input_vel = 1
        odrv0.axis0.controller.input_vel = -1

    if movement == "forward":
        odrv0.axis1.controller.input_vel = input_velocity
        odrv0.axis0.controller.input_vel = -input_velocity

    elif movement == "backward":
        odrv0.axis1.controller.input_vel = -input_velocity
        odrv0.axis0.controller.input_vel = input_velocity

    elif movement == "left":
        odrv0.axis1.controller.input_vel = input_velocity
        odrv0.axis0.controller.input_vel = -input_velocity/2

    elif movement == "right":
        odrv0.axis1.controller.input_vel = input_velocity/2
        odrv0.axis0.controller.input_vel = -input_velocity
    else:
        #No movement
        odrv0.axis1.controller.input_vel = 0
        odrv0.axis0.controller.input_vel = 0

    if peopleDetect:
        print("Detecting people")
        # People or Stop sign detected
        if 4 in classIDs or 8 in classIDs:
            print("Stop")
            odrv0.axis1.controller.input_vel = 0
            odrv0.axis0.controller.input_vel = 0

        # Slow or Speed sign or Hump or Pedestrain Sign detected
        elif 6 in classIDs or 7 in classIDs or 1 in classIDs or 3 in classIDs:
            print("Slow")
            odrv0.axis1.controller.input_vel = 0.8
            odrv0.axis0.controller.input_vel = -0.8

def steer():
    # global minBrozimuth, maxBrozimuth, brozimuth, classIDs, pot_value
    pot_value = map_value (chan1.value, 0, 26230, 0, 1023)
    steering_angle = map_value(pot_value, 0 , 1023, -40 ,40)

    steerLeftLimit = 10
    steerRightLimit = -10
    
    #Middle/Straight values
    steerLeftTurnLimit = -6
    steerRightTurnLimit = -3

    if auto:
        print("auto: ", steering_angle)

        # if minBrozimuth <= steering_angle <= maxBrozimuth:  #If steering angle is within turn range

        #If exceed limit, set to limit
        #Limit for auto
        if brozimuth > steerLeftLimit:
            brozimuth = steerLeftLimit
        elif brozimuth < steerRightLimit:
            brozimuth = steerRightLimit

        print("In brozimuth range")
        pwm.value = True  # Set GPIO12 high
        steering.value = True  # Set GPIO13 low

        if minBrozimuth <= steering_angle <= maxBrozimuth:
            print("In steering range")
            pwm.value = False
            steering.value = False

            if within_tolerance:    #within_tolerance to turn
                if steerLeftTurnLimit <= steering_angle <= steerRightTurnLimit:
                    print("In steering range")
                    pwm.value = False
                    steering.value = False
                if steering_angle <= steerLeftTurnLimit:  # Steer Left
                    pwm.value = True
                    steering.value = False
                    if pot_value is not None:
                        print(f"Steering Left: Potentiometer Value: {int(steering_angle)}")
                elif steering_angle > steerRightTurnLimit:    #Steer right
                    pwm.value = True
                    steering.value = True
                    if pot_value is not None:
                        print(f"Steering Right: Potentiometer Value: {int(steering_angle)}")

        else:
            if steering_angle <= brozimuth:  # Steer Left
                pwm.value = True
                steering.value = False
                if pot_value is not None:
                    print(f"Steering Left: Potentiometer Value: {int(steering_angle)}")
            elif steering_angle > brozimuth:    #Steer right
                pwm.value = True
                steering.value = True
                if pot_value is not None:
                    print(f"Steering Right: Potentiometer Value: {int(steering_angle)}")
            else:   #Steer right limit
                pwm.value = False
                steering.value = False
                if pot_value is not None:
                    print(f"No Steering: Potentiometer Value: {int(steering_angle)}")


    elif not auto:
        print("Manual")
        # 'a' is pressed
        if keyboard.is_pressed('a') or 2 in classIDs and steering_angle is not None:
            print("Steer left", steering_angle)
            if steering_angle <= steerLeftLimit:  # Steer Left
                pwm.value = True
                steering.value = False
                if pot_value is not None:
                    print(f"Steering Left: Potentiometer Value: {int(steering_angle)}")
            else:   #Steer right limit
                pwm.value = False
                steering.value = False
                if pot_value is not None:
                    print(f"No Steering: Potentiometer Value: {int(steering_angle)}")
        
        elif keyboard.is_pressed('d') or 5 in classIDs and steering_angle is not None:
            print("Steer right", steering_angle)
            if steering_angle >= steerRightLimit:    #Steer right
                pwm.value = True
                steering.value = True
                if pot_value is not None:
                    print(f"Steering Right: Potentiometer Value: {int(steering_angle)}")
            else:   #Steer left limit
                pwm.value = False
                steering.value = False
                if pot_value is not None:
                    print(f"No Steering: Potentiometer Value: {int(steering_angle)}")

        else: # Joystick IDLE
            pwm.value = False
            steering.value = False
            if pot_value is not None:
                print(f"No Steering: Potentiometer Value: {int(steering_angle)}")
    
    else:
        pwm.value = False
        steering.value = False
        if pot_value is not None:
            print(f"No Steering: Potentiometer Value: {int(steering_angle)}")
    
    if keyboard.is_pressed('esc'):
        pwm.value = False
        steering.value = False

# Toggle between manual and auto
def toggle_control():
    global auto, input_velocity, steering_angle, speedMode
    o_pressed_time = None  # Track when 'o' is first pressed

    while True:
        if keyboard.is_pressed('o'):    #press o or if held longer than 1 sec
            if o_pressed_time is None:
                o_pressed_time = time.time()  # Start tracking time
            elif time.time() - o_pressed_time > 1:  # Check if held for 1 second
                auto = not auto
                print(f"Switched to {'Auto' if auto else 'Manual'} mode")
            else:
                o_pressed_time = None  # Track when 'o' is first pressed


        if keyboard.is_pressed('space'):
            speedMode = not speedMode

            if speedMode:
                input_velocity = 2
                print(f"Increased velocity to: {input_velocity}")
            else:
                input_velocity = 1
                print(f"Velocity to: {input_velocity}")

        if auto:
            auto_control()
        else:
            manual_control()

        time.sleep(0.1)

exit_flag = False

# Main function
def main():
    global exit_flag, recording, out_annotated, img_counter, vid_counter,frame1,steering_angle, last_voltage_print_time
    # Main event loop
    while not exit_flag:
        # Print the ODrive voltage at a 1-second interval
        current_time = time.time()
        if current_time - last_voltage_print_time >= 0.3:  # 0.3 second interval
            print("ODrive Voltage:", odrv0.vbus_voltage)
            last_voltage_print_time = current_time        # Poll for pygame events

        detection(cap)

        camera(cap1)

        # Check for 'q' key in pygame to exit
        if keyboard.is_pressed('q'):
            print("Resetting")
            odrv0.axis1.controller.input_vel = 0
            odrv0.axis0.controller.input_vel = 0
            if odrv0.axis0.controller.input_vel == 0 or odrv0.axis1.controller.input_vel == 0:
                odrv0.axis1.requested_state = 3  # Set ODrive to idle state
                odrv0.axis0.requested_state = 3  # Set ODrive to idle state

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
    detection_thread.join()
    

async def FasttAPIServer():
    print("FastAPIServer")
    # Start the FastAPI server
    config = Config(app, host="127.0.0.1", port=8765)
    server = Server(config)
    await server.serve()


if __name__ == '__main__':
    detection_thread = threading.Thread(target=main, daemon=True)
    detection_thread.start()

    toggle_thread = threading.Thread(target=toggle_control, daemon=True)
    toggle_thread.start()

    try:
        asyncio.run(FasttAPIServer())
    except KeyboardInterrupt:
        print("Stopping all threads...")
        # Properly handle thread shutdown
        detection_thread.join()
        toggle_thread.join()
        print("All threads stopped. Exiting application.")
