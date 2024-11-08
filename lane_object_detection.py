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

# Global variable to track drivable area detection and object detection
drivable_area_detected = False  

run_detection = True
Auto_driving = True
Steering = False

#object detection class_id
class_ids=[]


# glboal cap for camera
cap = cv2.VideoCapture(6) #Lane
cap1 = cv2.VideoCapture(4) #Object
# Load the model
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load('unet_model_with_class_weights.pth'))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

modely = YOLO("best.pt")


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
out_combined = None

# Initialize pygame
pygame.init()

# Define window size (To capture events)
screen = pygame.display.set_mode((640,480))


# Preprocessing function for camera frame
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame / 255.0).permute(2, 0, 1).float().unsqueeze(0)  # Add batch dimension
    return frame_tensor

def detection(cap, cap1, run_detection=True):
    #global variables to pass
    global drivable_area_detected, class_ids, annotated_frame, combined_image

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    if True:
        ret, frame = cap.read()
        ret1, frame1 = cap1.read()  # Capture frame-by-frame

        if not ret:
            print("Failed to grab frame")
            
        if not ret1:
            print("Failed to grab frame1")
            
        # Run YOLO detection if enabled
        if run_detection:
            results = modely(frame1, conf=0.5, verbose=False, device=0)  # Running detection
            annotated_frame = results[0].plot()  # Annotate frame with bounding boxes and labels
            # Process and print class names for each detection
            for result in results:
                class_ids = result.boxes.cls.tolist()
                class_names = [modely.names[int(id)] for id in class_ids]  # Map IDs to names
            
            # Count occurrences of each class name
            class_counts = Counter(class_names)
            for class_name, count in class_counts.items():
                print(f"{count} {class_name} detected.")
        else:
            results = modely(frame1, conf=0.5, verbose=False, device=0)  # Running detection
            annotated_frame = results[0].plot()
        # Display the annotated frame
        cv2.imshow('YOLOv8 Detection', annotated_frame)

        # Preprocess the frame
        frame = cv2.flip(frame, -1)

        # Preprocess the frame
        frame_tensor = preprocess_frame(frame).to(device)

        # Get model prediction
        with torch.no_grad():
            output = model(frame_tensor)
            predicted_mask = torch.sigmoid(output)
            predicted_mask = (predicted_mask > 0.7).float().squeeze().cpu().numpy()

        # Resize predicted mask to match original frame
        predicted_mask_resized = cv2.resize(predicted_mask, (640, 480), interpolation=cv2.INTER_NEAREST)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(predicted_mask_resized.astype(np.uint8), connectivity=8)

        # Display lanes only if the area exceeds the threshold
        if num_labels > 1:  # num_labels includes the background as label 0
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip background
            largest_mask = (labels == largest_label).astype(np.uint8)

            # Smooth the largest mask with morphological operations
            kernel = np.ones((5, 5), np.uint8)
            largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_CLOSE, kernel)  # Close small gaps
            largest_mask = cv2.GaussianBlur(largest_mask, (5, 5), 0)  # Apply Gaussian blur

            # Convert largest mask to overlay on the frame
            overlay = np.zeros_like(frame)
            overlay[largest_mask > 0.5] = [0, 255, 0]  # Green for the detected largest area

            # Combine original frame with overlay
            combined_image = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            # Display the result
            cv2.imshow('Smoothed Largest Drivable Area Detection', combined_image)
            drivable_area_detected = True  # Set to True when a drivable area is detected
        else:
            # Display the original frame if no drivable area is detected
            cv2.imshow('Smoothed Largest Drivable Area Detection', frame)
            drivable_area_detected = False  # Reset to False when no drivable area is detected
            return combined_image, annotated_frame

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

# Connect to ODrive
odrv0 = connect_to_odrive()
# Display drive voltage
print("ODrive Voltage:", odrv0.vbus_voltage)
odrv0.axis0.controller.config.control_mode = 2
odrv0.axis0.controller.config.input_mode = 2
odrv0.axis0.requested_state = 8  # Reactivate ODrive
odrv0.axis1.requested_state = 8 # Reactivate ODrive
calibration(odrv0, duration=20)

def Manual_drive(keys):
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
            odrv0.axis1.controller.input_vel = 0 #Stop WHeels before moving backwards
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
#     global drivable_area_detected  # Declare as global to access the flag

#     host = '192.168.1.104'  # Accept connections from any IP address
#     port = 5000  # Port to listen on

#     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server_socket.bind((host, port))
#     server_socket.listen(1)  # Accept only one connection for simplicity

#     print("Waiting for connection from ESP32...")
#     conn, address = server_socket.accept()  # Accept new connection
#     print("Connection from:", address)

#     try:
#         while True:
#             # Receive data
#             data = conn.recv(1024).decode()
#             if not data:
#                 break
#             print("Received from ESP32:", data)

#             if data == "depart":
#                 print("Starting motor")
#                 if not drivable_area_detected:  # Check if a drivable area was detected
#                     lane_detection()
                    
#                 else:
#                     odrv0.axis0.requested_state = 8  # Activate ODrive axis 0
#                     odrv0.axis1.requested_state = 8  # Activate ODrive axis 1

#             elif data == "arrive":
#                 print("Stopping motor...")
#                 odrv0.axis0.requested_state = 3  # Set ODrive axis 0 to idle
#                 odrv0.axis1.requested_state = 3  # Set ODrive axis 1 to idle


#             # Send response
#             response = "Hello from Jetson"
#             conn.send(response.encode())
#     except Exception as e:
#         print("Error:", e)
#     finally:
#         conn.close()  # Close the connection
def main():
    # global exit variable
    global exit_flag, run_detection, Auto_driving, recording, out_annotated, out_combined, img_counter, vid_counter
    # Start lane detection in a separate thread or process

    # odrv0.axis0.requested_state = 8  # Activate ODrive axis 0
    # odrv0.axis1.requested_state = 8  # Activate ODrive axis 1
    # odrv0.axis1.controller.input_vel = 2
    # odrv0.axis0.controller.input_vel = -2
    # Main event loop
    while not exit_flag:
        import threading
        lane_detection_thread = threading.Thread(target=detection(cap, cap1, run_detection))
        lane_detection_thread.start()
        # Poll for pygame events
        pygame.event.pump()
        keys = pygame.key.get_pressed()

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
                odrv0.axis1.controller.input_vel = 0
                odrv0.axis0.controller.input_vel = 0
                continue

        elif not Auto_driving:
            Manual_drive(keys)

        # If recording, write the frame to the video output file
        if recording:
            out_annotated.write(annotated_frame)
            out_combined.write(combined_image)

        if keys[pygame.K_m]:
            Auto_driving = not Auto_driving # Toggle Auto Driving
            run_detection = not run_detection # Toggle run_detection
            print (f"Auto Driving is now : [{Auto_driving}]")
            print (f"Object Detection is now: [{run_detection}]")


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
                out_combined = cv2.VideoWriter(vid_name_combined, fourcc, 20.0, 
                                               (combined_image.shape[1], combined_image.shape[0]))
                recording = True
            else:
                print("Recording stopped.")
                recording = False
                out_annotated.release()  # Stop recording and release the output file for annotated frames
                out_combined.release()   # Stop recording and release the output file for combined frames
                print(f"{vid_name_annotated} and {vid_name_combined} saved!")
                vid_counter += 1

        # Image capture functionality with 't' key for both frames
        elif keys[pygame.K_t]:
            img_name_annotated = os.path.join(output_image_dir, f"object_frame_{img_counter}.png")
            img_name_combined = os.path.join(output_image_dir, f"lane_frame_{img_counter}.png")
            cv2.imwrite(img_name_annotated, annotated_frame)  # Save YOLO-detected frame
            cv2.imwrite(img_name_combined, combined_image)    # Save lane-detected frame
            print(f"Images saved: {img_name_annotated} and {img_name_combined}")
            img_counter += 1

        # 'l' key to see status
        elif keys[pygame.K_l]: 
            print("Status:")
            print ("Auto:",[{Auto_driving}])
            print ("Detection:",[{run_detection}])
            print ("recording:",[{recording}])


        cv2.waitKey(1)
    #Quit all components
    pygame.quit()
    cap.release()
    cv2.destroyAllWindows()  # Close windows after exiting
    print("Exiting all components.")

exit_flag = False

if  __name__ == '__main__':

    main()
