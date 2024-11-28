# main.py
import socket
import odrive
import torch
import cv2
import numpy as np
from FINAL_UNET import UNet  # Import the UNet class from unet.py
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


# Function to map chan1 value to a specific range (e.g., 10 to 100)
def map_potentiometer_value(value, input_min=0, input_max=26230, output_min=0, output_max=1023):
    return ((value - input_min) * (output_max - output_min)) / (input_max - input_min) + output_min

#Initialise GPIO Pins
pwm = digitalio.DigitalInOut(board.D12)
steering = digitalio.DigitalInOut(board.D13)
pwm.direction = digitalio.Direction.OUTPUT
steering.direction = digitalio.Direction.OUTPUT

#Initialise actuator IDLE 
pwm.value = True  # Set GPIO12 high
steering.value = True  # Set GPIO13 high

# Global variable to track drivable area detection and object detection

run_detection = True
Auto_driving = True
Steering = False
esp = True
#object detection class_id
class_ids=[]

# global cap for camera
cap = DepthCamera()

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

#Initialise start mode
start_mode = True

# Initialize pygame
pygame.init()

# This is a simple class that will help us print to the screen.
# It has nothing to do with the joysticks, just outputting the
# information.
class TextPrint:
    def __init__(self):
        self.reset()
        self.font = pygame.font.Font(None, 25)

    def tprint(self, screen, text):
        text_bitmap = self.font.render(text, True, (0, 0, 0))
        screen.blit(text_bitmap, (self.x, self.y))
        self.y += self.line_height

    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 15

    def indent(self):
        self.x += 10

    def unindent(self):
        self.x -= 10

#gets centre of bounding bvox
def get_center_of_bbox(box):
    """ Calculate the center point of the bounding box """
    x_center = int((box[0] + box[2]) / 2)
    y_center = int((box[1] + box[3]) / 2)
    return (x_center, y_center)


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

                    cv2.circle(annotated_frame, center_point, 5, (0, 0, 255), -1)
                    distance = success[center_point[1], center_point[0]]
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

        # Display the annotated frame
        cv2.imshow('YOLOv8 Detection', annotated_frame)

# Connect to ODrive
odrv0 = connect_to_odrive()
last_voltage_print_time = time.time()  # Initialize the timer

# Display drive voltage
print("ODrive Voltage:", odrv0.vbus_voltage)
odrv0.axis0.controller.config.control_mode = 2 # Vel Mode
odrv0.axis0.controller.config.input_mode = 2 #Vel Mode
odrv0.axis0.requested_state = 8  # Start Motor
odrv0.axis1.requested_state = 8 # Start Motor

def Manual_drive(joysticks, screen, text_print, clock):
    # global exit variable
    global exit_flag, recording, out_annotated, img_counter, vid_counter, Steering, start_mode,run_detection, Auto_driving
    try:
        pot_value = map_potentiometer_value(chan1.value)
    except OSError as e:
        print(f"Error reading potentiometer: {e}")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_flag = True  # Flag that we are done so we exit this loop.

        if event.type == pygame.JOYBUTTONDOWN:
            # Square button
            if event.button == 3:
                start_mode = not start_mode
                #Start motor
                if start_mode:
                    print("Starting...")
                    odrv0.axis0.requested_state = 8  # Start Motor
                    odrv0.axis1.requested_state = 8 # Start Motor

                #Motor Idle
                if start_mode is False: 
                    print("Resetting")
                    odrv0.axis1.controller.input_vel = 0
                    odrv0.axis0.controller.input_vel = 0
                    if odrv0.axis0.controller.input_vel == 0 or odrv0.axis1.controller.input_vel == 0:
                        odrv0.axis1.requested_state = 3  # Set ODrive to idle state
                        odrv0.axis0.requested_state = 3  # Set ODrive to idle state

            # Circle Button
            if event.button == 1:
                print("Status:")
                print ("recording:",[{recording}])

            # Triangle Button
            if event.button == 2:
                if not recording:
                    print("Recording started for both annotated and combined frames...")
                    # Set up video writers for both annotated and combined frames
                    vid_name_annotated = os.path.join(output_video_dir, f"object_video_{vid_counter}.mp4")
                    vid_name_combined = os.path.join(output_video_dir, f"lane_video_{vid_counter}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                    out_annotated = cv2.VideoWriter(vid_name_annotated, fourcc, 20.0, (annotated_frame.shape[1], annotated_frame.shape[0]))
                    recording = True
                else:
                    print("Recording stopped.")
                    recording = False
                    out_annotated.release()  # Stop recording and release the output file for annotated frames
                    print(f"{vid_name_annotated} and {vid_name_combined} saved!")
                    vid_counter += 1
            
            # Triangle Button
            if event.button == 0:
                print ("Image Captured")
                img_name_annotated = os.path.join(output_image_dir, f"object_frame_{img_counter}.png")
                img_name_combined = os.path.join(output_image_dir, f"lane_frame_{img_counter}.png")
                cv2.imwrite(img_name_annotated, annotated_frame)  # Save YOLO-detected frame
                print(f"Images saved: {img_name_annotated} and {img_name_combined}")
                img_counter += 1

            #Left axis Motion Button
            if event.button == 11:
                odrv0.axis1.controller.input_vel = 0
                odrv0.axis0.controller.input_vel = 0
                print ('STOP')

            if event.button == 5:
                Auto_driving = not Auto_driving
                print (f'Auto driving: {Auto_driving}')
            # Button index for top-right (R1/RB)
            if event.button == 10:  
                print("Exiting...")
                odrv0.axis1.controller.input_vel = 0
                odrv0.axis1.requested_state = 1  # Set ODrive to idle state
                odrv0.axis0.controller.input_vel = 0
                odrv0.axis0.requested_state = 1  # Set ODrive to idle state
                exit_flag = True       
                break
                
        if event.type == pygame.JOYBUTTONUP:
            print("Joystick button released.")
        # Handle hotplugging
        if event.type == pygame.JOYDEVICEADDED:
            # This event will be generated when the program starts for every
            # joystick, filling up the list without needing to create them manually.
            joy = pygame.joystick.Joystick(event.device_index)
            joysticks[joy.get_instance_id()] = joy
            print(f"Joystick {joy.get_instance_id()} connencted")

        if event.type == pygame.JOYDEVICEREMOVED:
            del joysticks[event.instance_id]
            print(f"Joystick {event.instance_id} disconnected")

        # Axis Motion
        if event.type == pygame.JOYAXISMOTION:
                # Handle joystick axis motion
            if event.instance_id in joysticks:
                joy = joysticks[event.instance_id]            
                x_axis = joy.get_axis(0)  # Horizontal axis
                y_axis = joy.get_axis(1)  # Vertical axis
                AXIS_THRESHOLD = 0.4 # Adjust
                """Determine the joystick direction based on axis values."""
                if abs(x_axis) < AXIS_THRESHOLD and abs(y_axis) < AXIS_THRESHOLD: # Steering Idle
                    pwm.value = True
                    steering.value = True
                    if pot_value is not None:
                        print(f"No Steering: Potentiometer Value: {pot_value}")
                    Steering = False

                if y_axis < -AXIS_THRESHOLD:   # Move Forward
                    odrv0.axis1.controller.input_vel = -2
                    odrv0.axis0.controller.input_vel = -2   
                    print ('FORWARD')

                if y_axis > AXIS_THRESHOLD: # Move BackWard
                    odrv0.axis1.controller.input_vel = 2
                    odrv0.axis0.controller.input_vel = 2                
                    print ('BACKWARD')

                if x_axis < -AXIS_THRESHOLD and pot_value <= 800:  # Steer Left
                    pwm.value = False
                    steering.value = True
                    if pot_value is not None:
                        print(f"Steering Left: Potentiometer Value: {pot_value}")
                    Steering = True
                    print ("Left")

                if x_axis > AXIS_THRESHOLD and pot_value >= 300:  # Steer Right
                    pwm.value = False
                    steering.value = False
                    if pot_value is not None:
                        print(f"Steering Right: Potentiometer Value: {pot_value}")
                    Steering = True
                    print ("Right")            
    # Drawing step
    # First, clear the screen to white. Don't put other drawing commands
    # above this, or they will be erased with this command.
    screen.fill((255, 255, 255))
    text_print.reset()

    # # If recording, write the frame to the video output file
    # if recording:
    #     out_annotated.write(annotated_frame)

    # Get count of joysticks.
    joystick_count = pygame.joystick.get_count()

    text_print.tprint(screen, f"Number of joysticks: {joystick_count}")
    text_print.indent()

    # For each joystick:
    for joystick in joysticks.values():
        jid = joystick.get_instance_id()

        text_print.tprint(screen, f"Joystick {jid}")
        text_print.indent()

        # Get the name from the OS for the controller/joystick.
        name = joystick.get_name()
        text_print.tprint(screen, f"Joystick name: {name}")

        guid = joystick.get_guid()
        text_print.tprint(screen, f"GUID: {guid}")

        power_level = joystick.get_power_level()
        text_print.tprint(screen, f"Joystick's power level: {power_level}")

        # Usually axis run in pairs, up/down for one, and left/right for
        # the other. Triggers count as axes.
        axes = joystick.get_numaxes()
        text_print.tprint(screen, f"Number of axes: {axes}")
        text_print.indent()

        for i in range(axes):
            axis = joystick.get_axis(i)
            text_print.tprint(screen, f"Axis {i} value: {axis:>6.3f}")
        text_print.unindent()

        buttons = joystick.get_numbuttons()
        text_print.tprint(screen, f"Number of buttons: {buttons}")
        text_print.indent()

        for i in range(buttons):
            button = joystick.get_button(i)
            text_print.tprint(screen, f"Button {i:>2} value: {button}")
        text_print.unindent()

        hats = joystick.get_numhats()
        text_print.tprint(screen, f"Number of hats: {hats}")
        text_print.indent()

        # Hat position. All or nothing for direction, not a float like
        # get_axis(). Position is a tuple of int values (x, y).
        for i in range(hats):
            hat = joystick.get_hat(i)
            text_print.tprint(screen, f"Hat {i} value: {str(hat)}")
        text_print.unindent()

        text_print.unindent()

        
    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

    # Limit to 30 frames per second.
    clock.tick(30)

    cv2.waitKey(1)

    # return pot_value
        
def main():
    # global exit variable
    global exit_flag, recording, out_annotated, img_counter, vid_counter, last_voltage_print_time, run_detection, Auto_driving
    # Start lane detection in a separate thread or process
    # Main event loop
        # Set the width and height of the screen (width, height), and name the window.
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("Joystick Info")

    # Used to manage how fast the screen updates.
    clock = pygame.time.Clock()

    # Get ready to print.
    text_print = TextPrint()

    # This dict can be left as-is, since pygame will generate a
    # pygame.JOYDEVICEADDED event for every joystick connected
    # at the start of the program.
    joysticks = {}
    while not exit_flag:
        # Print the ODrive voltage at a 1-second interval
        current_time = time.time()
        if current_time - last_voltage_print_time >= 0.3:  # 0.3 second interval
            print("ODrive Voltage:", odrv0.vbus_voltage)
            last_voltage_print_time = current_time        # Poll for pygame events


        detection(cap)
        
        if Auto_driving:
            # Control motors based on object detections
            if 4 in class_ids or 8 in class_ids:
                print("Stop")
                # Trigger joystick vibration
                for joystick in joysticks.values():
                    try:
                        # Set vibration intensity (low and high frequency) and duration
                        joystick.rumble(0.5, 0.5, 1000)  # 0.5 intensity, 1-second duration
                    except NotImplementedError:
                        print(f"Joystick {joystick.get_instance_id()} does not support vibration.")
        
                run_detection = False
                Auto_driving = False
                esp = False
                odrv0.axis1.controller.input_vel = 0
                odrv0.axis0.controller.input_vel = 0
                continue
 
            elif 6 in class_ids or 7 in class_ids:
                print("Slow")
                odrv0.axis1.controller.input_vel = -1
                odrv0.axis0.controller.input_vel = -1

            if 1 in class_ids or 3 in class_ids:
                print("Pedestrain")
                odrv0.axis1.controller.input_vel = -1
                odrv0.axis0.controller.input_vel = -1
                if 4 in class_ids:
                    print("Stop")
                    odrv0.axis1.controller.input_vel = 0
                    odrv0.axis0.controller.input_vel = 0
                else:
                    print("Continue")
                    odrv0.axis1.controller.input_vel = -2
                    odrv0.axis0.controller.input_vel = -2

        elif not Auto_driving:
            Manual_drive(joysticks, screen, text_print, clock)

            pot_value = Manual_drive(joysticks, screen, text_print, clock)

            if pot_value == None:
                continue

    #Quit all components
    pygame.quit()
    cap.release()
    cv2.destroyAllWindows()  # Close windows after exiting
    print("Exiting all components.")

exit_flag = False

if  __name__ == '__main__':

    main()