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

# Initialize pygame
pygame.init()

# Define window size (To capture events)
screen = pygame.display.set_mode((640,480))

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

# Calibration for odrive wheels
def calibration(odrv, duration=20):
    """
    Move axis0 and axis1 of the given ODrive instance in a defined pattern
    for a specified duration.
    
    Parameters:
        odrv: ODrive instance
        duration: Total duration of the movement in seconds
    """
    # Define the key states
    states = [(2, -2), (0, 0), (-2, 2)]
    state_duration = duration / len(states)  # Duration for each state

    start_time = time.time()
    
    while time.time() - start_time < duration:
        elapsed_time = time.time() - start_time

        # Determine the current state based on elapsed time
        current_state_index = int(elapsed_time // state_duration) % len(states)
        input_vel = states[current_state_index]

        # Set input velocities for the axes
        odrv.axis0.controller.input_vel = input_vel[1]  # Axis 0
        odrv.axis1.controller.input_vel = input_vel[0]  # Axis 1
        time.sleep(2)  # Small delay to prevent busy waiting

    # Set both axes to zero velocity at the end
    odrv.axis0.controller.input_vel = 0
    odrv.axis1.controller.input_vel = 0

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

# def connect_to_esp():
#     global conn  # Declare as global to access the flag

#     host = '192.168.230.167'  # Accept connections from any IP address
#     port = 8888  # Port to listen on

#     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server_socket.bind((host, port))
#     server_socket.listen(1)  # Accept only one connection for simplicity

#     print("Waiting for connection from ESP32...")
#     conn, address = server_socket.accept()  # Accept new connection
#     print("Connection from:", address)

# Connect to ODrive and ESP
odrv0 = connect_to_odrive()
# connect_to_esp()

# Display drive voltage
print("ODrive Voltage:", odrv0.vbus_voltage)
odrv0.axis0.controller.config.control_mode = 2 # Vel Mode
odrv0.axis0.controller.config.input_mode = 2 #Vel Mode
odrv0.axis0.requested_state = 8  # Start Motor
odrv0.axis1.requested_state = 8 # Start Motor

calibration(odrv0, duration=20)

def Manual_drive(keys):
        global Steering
        # 'e' key to start motor
        if keys[pygame.K_e]:  
            print("Starting...")
            odrv0.axis0.requested_state = 8  # Start Motor
            odrv0.axis1.requested_state = 8 # Start Motor

        # 'w' key to move forward
        if keys[pygame.K_w]:
            odrv0.axis1.controller.input_vel = 0 #Stop Wheels before moving forward
            odrv0.axis0.controller.input_vel = 0 

            if odrv0.axis0.controller.input_vel == 0:
                odrv0.axis1.controller.input_vel = 2
                odrv0.axis0.controller.input_vel = -2
                print ('FORWARD')
        # 's' key to move backwards
        if keys[pygame.K_s]:
            odrv0.axis1.controller.input_vel = 0 #Stop Wheels before moving backwards
            odrv0.axis0.controller.input_vel = 0
            if odrv0.axis0.controller.input_vel == 0:
                odrv0.axis1.controller.input_vel = -2
                odrv0.axis0.controller.input_vel = 2
                print ('BACKWARD')

        # 'shift' key to brake
        mods = pygame.key.get_mods()  # Get current modifier key states
        if mods & pygame.KMOD_SHIFT:
            odrv0.axis1.controller.input_vel = 0
            odrv0.axis0.controller.input_vel = 0
            print ('STOP')

        #'a' is pressed
        if keys[pygame.K_a]:  # Steer Left
            # GPIO.output(pwm, GPIO.LOW)
            # GPIO.output(steering, GPIO.HIGH)
            print ('LEFT')
            Steering = True

        #'d' is pressed
        if keys[pygame.K_d]:  # Steer Right
            # GPIO.output(pwm, GPIO.LOW)
            # GPIO.output(steering, GPIO.LOW) 
            print('RIGHT')
            Steering = True

        if not keys[pygame.K_a] and not keys[pygame.K_d]:  # Joystick IDLE
            # GPIO.output(pwm, GPIO.HIGH)
            # GPIO.output(steering, GPIO.HIGH)
            Steering = False

# def server_program():
#     # Receive data
#     data = conn.recv(1024).decode()
#     if not data and esp:
#         print("Received from ESP32:", data)

#     if data == "test" and esp:
#         print("Calibration 50/100")

#     elif data == "test1" and esp:
#         print("Calibration 100/100")

#     elif data == "start" and esp:
#         print("Starting motor")
#         odrv0.axis0.requested_state = 8  # Activate ODrive axis 0
#         odrv0.axis1.requested_state = 8  # Activate ODrive axis 1
    
#     elif data =="forward" and esp:
#         odrv0.axis1.controller.input_vel = 0 #Stop Wheels before moving forward
#         odrv0.axis0.controller.input_vel = 0  
#         if odrv0.axis0.controller.input_vel == 0:
#             odrv0.axis1.controller.input_vel = 2
#             odrv0.axis0.controller.input_vel = -2
#             print ('FORWARD')

#     elif data =="backward" and esp:
#         odrv0.axis1.controller.input_vel = 0 #Stop WHeels before moving backwards
#         odrv0.axis0.controller.input_vel = 0

#         if odrv0.axis0.controller.input_vel == 0:
#             odrv0.axis1.controller.input_vel = -2
#             odrv0.axis0.controller.input_vel = 2
#             print ('BACKWARD')

#     elif data == "stop" and esp:
#         print("Stopping motor...")
#         odrv0.axis1.controller.input_vel = 0 #Stop WHeels before moving backwards
#         odrv0.axis0.controller.input_vel = 0
    
#     else:
#         e = bool
#         Exception == e
#         print("Error:", e)

#     # Send response
#     response = "Hello from Jetson"
#     conn.send(response.encode())
        
def main():
    # global exit variable
    global exit_flag, run_detection, Auto_driving, recording, out_annotated, img_counter, vid_counter, esp
    # Start lane detection in a separate thread or process
    # Main event loop
    while not exit_flag:
        # Poll for pygame events
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        #Threading for detection and server_program
        import threading
        object_detection_thread = threading.Thread(target=detection(cap))
        object_detection_thread.start()
        # server_program_thread = threading.Thread(target=server_program)
        # server_program_thread.start()

        # Check for 'q' key in pygame to exit
        if keys[pygame.K_q]:
            print("Resetting")
            odrv0.axis1.controller.input_vel = 0
            odrv0.axis1.requested_state = 3  # Set ODrive to idle state
            odrv0.axis0.controller.input_vel = 0
            odrv0.axis0.requested_state = 3  # Set ODrive to idle state

        if Auto_driving:
            # Control motors based on object detections
            if 4 in class_ids or 8 in class_ids:
                print("Stop")
                run_detection = False
                Auto_driving = False
                esp = False
                odrv0.axis1.controller.input_vel = 0
                odrv0.axis0.controller.input_vel = 0
                continue
 
            elif 6 in class_ids or 7 in class_ids:
                print("Slow")
                odrv0.axis1.controller.input_vel = 1
                odrv0.axis0.controller.input_vel = -1

            if 1 in class_ids or 3 in class_ids:
                print("Pedestrain")
                odrv0.axis1.controller.input_vel = 1
                odrv0.axis0.controller.input_vel = -1
                if 4 in class_ids:
                    print("Stop")
                    odrv0.axis1.controller.input_vel = 0
                    odrv0.axis0.controller.input_vel = 0
                else:
                    print("Continue")
                    odrv0.axis1.controller.input_vel = 2
                    odrv0.axis0.controller.input_vel = -2

        elif not Auto_driving:
            Manual_drive(keys)

        # If recording, write the frame to the video output file
        if recording:
            out_annotated.write(annotated_frame)

        if keys[pygame.K_m]:
            Auto_driving = not Auto_driving # Toggle Auto Driving
            run_detection = not run_detection # Toggle run_detection
            esp = not esp #Toggle esp
            print (f"Auto Driving is now : [{Auto_driving}]")
            print (f"Object Detection is now: [{run_detection}]")
            print (f"ESP is now: [{esp}]")


        # Break the loop if 'ESC'
        elif keys[pygame.K_ESCAPE]:  # ESC to exit and restart motor
            print("Exiting...")
            odrv0.axis1.controller.input_vel = 0
            odrv0.axis1.requested_state = 3  # Set ODrive to idle state
            odrv0.axis0.controller.input_vel = 0
            odrv0.axis0.requested_state = 3  # Set ODrive to idle state
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

                out_annotated = cv2.VideoWriter(vid_name_annotated, fourcc, 20.0, 
                                                (annotated_frame.shape[1], annotated_frame.shape[0]))
                recording = True
            else:
                print("Recording stopped.")
                recording = False
                out_annotated.release()  # Stop recording and release the output file for annotated frames
                print(f"{vid_name_annotated} and {vid_name_combined} saved!")
                vid_counter += 1

        # Image capture functionality with 't' key for both frames
        elif keys[pygame.K_t]:
            img_name_annotated = os.path.join(output_image_dir, f"object_frame_{img_counter}.png")
            img_name_combined = os.path.join(output_image_dir, f"lane_frame_{img_counter}.png")
            cv2.imwrite(img_name_annotated, annotated_frame)  # Save YOLO-detected frame
            print(f"Images saved: {img_name_annotated} and {img_name_combined}")
            img_counter += 1

        # 'l' key to see status
        elif keys[pygame.K_l]: 
            print("Status:")
            print ("Auto:",[{Auto_driving}])
            print ("Detection:",[{run_detection}])
            print ("recording:",[{recording}])
            print ("ESP:",[{esp}])

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