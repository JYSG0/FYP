import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio
import pygame

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image
import odrive

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm

#YOLO
from ultralytics import YOLO
from collections import Counter
from realsense_depth import *
from ultralytics.utils.plotting import Annotator, colors

#JETSON PINS
import Jetson.GPIO as GPIO
import busio
import board
import digitalio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

#ESP
import keyboard

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

# # Initialize Pygame for PS4 controller vibration
# pygame.init()

# Store connected WebSocket clients (if needed for multi-client scenarios)
connected_clients = []

#Direction to turn
dir = None  
angleToTurn = None

#object detection class_id
class_ids=[]

run_detection = True

#Manual_mode
is_recording = False
manual_mode = True
start_mode = True
exit_flag = False
# Global variable to track drivable area detection and object detection
Steering = False
start_mode = True
speed_mode = False
lane_detect = False
input_velocity = 1  # Global variable to track velocity for motors


# Timing variables
last_press_time = 0
double_tap_threshold = 0.3  # Maximum time (seconds) between taps to count as a double-tap

prev_top_right =  prev_top_left = prev_top_center =  prev_bot_right =  prev_bot_left =  prev_bot_center = None


#Actuator
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

light1 = digitalio.DigitalInOut(board.D23)
light2 = digitalio.DigitalInOut(board.D26)
light1.direction = digitalio.Direction.OUTPUT
light2.direction = digitalio.Direction.OUTPUT
#Initial Low 
pwm.value = False  # Set GPIO12 high
steering.value = False  # Set GPIO13 low
light1.value = False  # Set GPIO12 high
light2.value = False  # Set GPIO13 low

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


        

def map_value(value, input_min, input_max, output_min, output_max):
    return ((value - input_min) * (output_max - output_min)) / (input_max - input_min) + output_min

def Driving(drivable,lane_position, odrv0, steering_angle, lane_detect): 
    if steering_angle is None:
        return


    if  lane_detect:
        if drivable:
            print("Drivable area detected.")
            odrv0.axis0.controller.input_vel = -1.0  # Move forward
            odrv0.axis1.controller.input_vel = 1.0  # Move forward
            cv2.putText(img_det, "Drivable Area", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            print("No drivable area detected. Stopping.")
            odrv0.axis0.controller.input_vel = 0  # Stop
            odrv0.axis1.controller.input_vel = 0  # Stop

        if lane_position == "left" and  lane_detect and steering_angle <= 15:
            print("Vehicle drifting left. Slowing down.")
            pwm.value = True
            steering.value = False
            light1.value = True
            light2.value = False  

            odrv0.axis0.controller.input_vel = -0.8
            odrv0.axis1.controller.input_vel = 1   # Stop


        elif lane_position == "right" and  lane_detect and steering_angle >= -8:
            print("Vehicle drifting right. Slowing down.")
            pwm.value = True
            steering.value = True
            light1.value = False
            light2.value = True  
            odrv0.axis0.controller.input_vel = -1
            odrv0.axis1.controller.input_vel = 0.8   # Stop

        if steering_angle >= 15 and lane_position == "left": #Steer Right Limit
            pwm.value = False
            steering.value = False
            print ("No Steering")          
        if steering_angle <= -8 and lane_position == "right": #Steer Left Limit
            pwm.value = False
            steering.value = False
            print ("No Steering")

        if lane_position == "center" or not lane_position: # Joystick IDLE
            if 1 <= steering_angle <= 15:
                print("Drifting Left, turning right")
                pwm.value = True
                steering.value = True
                light1.value = False
                light2.value = False
            if -8 <= steering_angle <= -1:
                print("Drifting right, turning Left")
                pwm.value = True
                steering.value = False

            if -1 <=steering_angle <= 1:
                pwm.value = False
                steering.value = True
            print("Vehicle centered in lane.")

            pwm.value = False
            steering.value = True
            if steering_angle is not None:
                print(f"No Steering: Potentiometer Value: {steering_angle}")


 
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

# Function to find the closest values on either side of center_x
def closest_values(arr, center, prev_values=None, threshold=100, y_axis=0.757):
    # Find the closest value greater than the center
    higher = arr[arr > center]
    lower = arr[arr < center]
    if len(higher) == 0 or len(lower) == 0:
        # if prev_values:
        #     prev_higher, prev_lower , prev_center = prev_values
        #     closest_higher = prev_higher
        #     closest_lower = prev_lower
        #     center = prev_center
        #     return closest_higher, closest_lower , center 
        return None, None, None
    

    closest_higher = higher.min()
    closest_lower = lower.max()

    center = (closest_higher + closest_lower) // 2
    # if prev_values:
    #     prev_higher, prev_lower , prev_center = prev_values
    #     print (prev_values)
    #     # Only check the differences if previous values are not None
    #     if prev_higher is not None and prev_lower is not None:

    #         # If difference in current value - previous value is higher, return previous value
    #         if abs(closest_higher - prev_higher) > threshold or abs(closest_lower - prev_lower) > threshold:

    #             closest_higher = prev_higher
    #             closest_lower = prev_lower
    #             center = prev_center

    #             return closest_higher, closest_lower , center

    return closest_higher, closest_lower , center

def curve_calculation(top_right, top_left,  bot_right, bot_left):
    left_curve = float((489-545) / (top_left - bot_left))
    right_curve = float((489-545) / (top_right - bot_right))
    print ("left_cruve: ", left_curve)
    print ("right_cruve: ", right_curve)
    return left_curve , right_curve
# Emergency stop function
def emergency_stop(odrv):
    try:
        odrv.axis0.controller.input_vel = 0
        odrv.axis1.controller.input_vel = 0
        odrv.axis0.requested_state = 1  # Set to idle state
        odrv.axis1.requested_state = 1
        print("Emergency Stop Activated!")
    except Exception as e:
        print(f"Error during emergency stop: {e}")


# Analyze segmentation results for drivable area and lane position
def process_segmentation(da_seg_mask, ll_seg_mask, img_det):
    global prev_top_right, prev_top_left, prev_top_center, prev_bot_right, prev_bot_left, prev_bot_center
    h, w = da_seg_mask.shape


    # Detect drivable area
    front_row = da_seg_mask[int(h * 0.8):, :]  # Bottom 20% of the frame
    drivable = np.sum(front_row) > 0

    # Draw the red dot
    dot_color = (255, 0, 255)  # BGR for red
    dot_radius = 5 # Radius of the dot
    dot_center = (640,315)
    cv2.line(img_det, dot_center, (640,720), dot_color, 2)  # -1 to fill the circle
    # print (f"x and y({vehicle_center , h})")
    
    # Analyze lane position
    resized_mask = cv2.resize(ll_seg_mask, (1280, 720), interpolation=cv2.INTER_NEAREST)
    h1, w1 = resized_mask.shape
    
    bottom_row = resized_mask[int(h1 * 0.514), :]  # Bottom row of the lane mask
    lane_pixels = np.where(bottom_row > 0)[0]
    top_row = resized_mask[int(h1 * 0.458), :]
    top_pixels = np.where(top_row>0,)[0]
    # cv2.line(img_det, (0,489), (1280, 489), (0,255,0), 2)
    # cv2.line(img_det, (0,545), (1280, 545), (0,255,0), 2)

    top_right, top_left, top_center = closest_values(top_pixels, 640, prev_values=(prev_top_right, prev_top_left, prev_top_center))
    bot_right, bot_left, bot_center = closest_values(lane_pixels, 640, prev_values=(prev_bot_right , prev_bot_left , prev_bot_center))


    
    if len(lane_pixels) >= 2:

        cv2.circle(img_det,(top_right, 330), 5 ,(255 ,0 ,255), -1)
        cv2.circle(img_det,(top_left, 330), 5 ,(255 ,0 ,255), -1)
        cv2.circle(img_det,(top_center, 330), 5 ,(255 ,0 ,255), -1)
        cv2.circle(img_det,(bot_right, 370), 5 ,(255 ,0 ,0), -1)
        cv2.circle(img_det,(bot_left, 370), 5 ,(0 ,0 ,255), -1)
        cv2.circle(img_det,(bot_center, 370), 5 ,(255 ,0 ,255), -1)
        print (f"bot_right:{bot_right}, bot_left:{bot_left}, bot_center:{bot_center}")
        print (f"top_right:{top_right}, top_left:{top_left}, top_center:{top_center}")

        print (f"prev_higher:{prev_bot_right}, prev_left:{prev_bot_left}, prev_center:{prev_bot_center}")
        # Save current values as previous values for the next iteration
        prev_top_right, prev_top_left, prev_top_center = top_right, top_left, top_center
        prev_bot_right, prev_bot_left, prev_bot_center = bot_right, bot_left, bot_center

        # # LANE TURNING
        # if (top_right and top_left and bot_right and bot_left) is not None:

        #     # #Calculate gradient of graphs
        #     # left_curve, right_curve = curve_calculation(top_right, top_left,  bot_right, bot_left)
        #     left_d = top_left - bot_left
        #     right_d = bot_right - top_right
        #     #Straight :D
        #     if bot_left < top_left and bot_right > top_right:

        #         print (f"left distance:{left_d} and right distance:{right_d}")
        
        #     #Left Turn :)
        #     if bot_left > top_left and bot_right > top_right:
        #         print ("Turn LEFTY")

        #     #Right Turn :O
        #     if bot_left < top_left and bot_right < top_right:
        #         print ("TURN RIGHTY")


        # LANE KEEPING
        if top_center is not None:
            if 640 > top_center:
                lane_position = "left"
                cv2.putText(img_det, "Lane Shift Left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            elif 640 < top_center:
                lane_position = "right"
                cv2.putText(img_det, "Lane Shift Right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            else:
                lane_position = "center"
                cv2.putText(img_det, "Good Lane Keeping", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            if top_right is None and top_left is not None:
                lane_position = "right"
                cv2.putText(img_det, "Lane Shift Right O_O", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if top_left is None and top_right is not None:
                lane_position = "left"
                cv2.putText(img_det, "Lane Shift Left O_O", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if (top_left and top_right) is None:
                lane_position = "SAVE ME"
                cv2.putText(img_det, "Emmanual :D please O_O", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
    else:
        lane_position = "unknown"

    return drivable, lane_position


def Manual_drive(odrv0, steering_angle, lane_position):
    global Steering, start_mode, input_velocity,speed_mode, lane_detect, light1, light2

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
        odrv0.axis1.controller.input_vel = input_velocity
        odrv0.axis0.controller.input_vel = -input_velocity
        if input_velocity == 1:

            #Sharp Left Turn
            if keyboard.is_pressed('a'):
                odrv0.axis0.controller.input_vel = -1
                odrv0.axis1.controller.input_vel = 0.5
            #Sharp Right Turn
            if keyboard.is_pressed('d'):
                odrv0.axis1.controller.input_vel = 1
                odrv0.axis0.controller.input_vel = -0.5

        if input_velocity == 2:
            #Sharp Left Turn
            if keyboard.is_pressed('a'):
                odrv0.axis0.controller.input_vel = -2
                odrv0.axis1.controller.input_vel = 1
            #Sharp Right Turn
            if keyboard.is_pressed('d'):
                odrv0.axis1.controller.input_vel = 2
                odrv0.axis0.controller.input_vel = -1
   
    if keyboard.is_pressed('s'):
        odrv0.axis1.controller.input_vel = -input_velocity
        odrv0.axis0.controller.input_vel = input_velocity
        print(f"BACKWARD at speed: {input_velocity}")

    if keyboard.is_pressed('o'):
        lane_detect = not lane_detect
        time.sleep(0.5)
        if lane_detect:  # Start lane_detection
            print("Detection On")

        else:  # Stop lane_detection
             print("Detection Off")
    if lane_position == "SAVE ME":
        light1.value = True
        light2.value = True
        lane_detect = False
    # 'shift' key to brake
    if keyboard.is_pressed('shift'):
        odrv0.axis1.controller.input_vel = 0
        odrv0.axis0.controller.input_vel = 0
        input_velocity = 1
        
        print ('STOP')


    if keyboard.is_pressed('a') and steering_angle is not None and steering_angle <= 15:  # Steer Left
        pwm.value = True
        steering.value = False
        print(f"Steering Left: Potentiometer Value: {int(steering_angle)}")

    if keyboard.is_pressed('d')  and steering_angle is not None and steering_angle >= -8:  # Steer Right
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
  
    if steering_angle is None:
        return
    return lane_detect
# Main detection and control loop
def detect(cfg, opt):
    #global variables to pass
    global is_recording, vid_counter, img_counter, img_det, vid_name , vid_name1 , out1
    distance = 0
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'demo')
    device = select_device(logger, opt.device)
    # # Set the width and height of the screen (width, height), and name the window.
    # screen = pygame.display.set_mode((640, 480))

    # Prepare output directory
    if os.path.exists(opt.save_dir):
        shutil.rmtree(opt.save_dir)
    os.makedirs(opt.save_dir)

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device).eval()

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size

    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors1 = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # Connect to ODrive
    odrv0 = connect_to_odrive()
    odrv0.axis0.requested_state = 8  # Closed-loop velocity control
    odrv0.axis1.requested_state = 8  # Closed-loop velocity control
    # Start inference
    for path, img, img_det, vid_cap, shapes in dataset:
        img = transform(img).to(device).unsqueeze(0)

        # Inference
        _, da_seg_out, ll_seg_out = model(img)

        # Process segmentation outputs
        _, _, height, width = img.shape
        da_seg_mask = torch.nn.functional.interpolate(da_seg_out, scale_factor=1, mode='bilinear').argmax(1).squeeze().cpu().numpy()
        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_out, scale_factor=1, mode='bilinear').argmax(1).squeeze().cpu().numpy()

        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)
        save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")


        if not is_recording:
            print("Recording started...")
            vid_name = os.path.join(output_video_dir, f"output_video_{vid_counter}.mp4")  # Set video name
            vid_name1 = os.path.join(output_video_dir, f"lane_video_{vid_counter}.mp4")  # Set video name
            
            # Define the codec and create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out1 = cv2.VideoWriter(vid_name1, fourcc, 20.0, (img_det.shape[1], img_det.shape[0]))
            is_recording = True
        
    # Read potentiometer value
        try:
            pot_value = map_value(chan1.value, 0, 26230, 0, 1023)
            steering_angle = map_value(pot_value, 0, 1023, -40, 40)
        except OSError as e:
            print(f"Error reading potentiometer: {e}")
            time.sleep(0.1)
            pot_value, steering_angle = None, None
 
    
        # Analyze results
        drivable, lane_position = process_segmentation(da_seg_mask, ll_seg_mask, img_det)
        Manual_drive(odrv0,steering_angle, lane_position)
        lane_detect = Manual_drive(odrv0,steering_angle, lane_position)

        Driving(drivable,lane_position, odrv0, steering_angle,lane_detect)
        # GPU memory usage
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # In MB
        reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 2)    # In MB
        free_memory = reserved_memory - allocated_memory                      # Free within reserved

        # print(f"Allocated Memory: {allocated_memory:.2f} MB")
        # print(f"Reserved Memory: {reserved_memory:.2f} MB")
        # print(f"Free Memory within Reserved: {free_memory:.2f} MB")
        cv2.imshow('YOLOP Inference', img_det)
        # Display the annotated frame
        if is_recording:
            out1.write(img_det)
        key = cv2.waitKey(1) & 0xFF  # Adjust delay based on video FPS

        # 's' key to capture an image
        if key == ord('s'):
            img_name = os.path.join(output_image_dir, f"opencv_frame_{img_counter}.png")  # Set image name
            img_name1 = os.path.join(output_image_dir, f"lanes_frame_{img_counter}.png")  # Set image name

            cv2.imwrite(img_name1, img_det)  # Write image file

            print(f"{img_name} written!")
            print(f"{img_name1} written!")

            img_counter += 1
        # Stop on key press
        if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
            emergency_stop(odrv0)
            break
        if exit_flag:
            break

    emergency_stop(odrv0)
    print("Inference completed.")
    print("Recording stopped.")
    is_recording = False
    print(f"{vid_name} written!")
    print(f"{vid_name1} written!")

    vid_counter += 1
    out1.release()
    # pygame.quit
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/End-to-end.pth', help='Path to model weights')
    parser.add_argument('--source', type=str, default='fieldestv2 - Made with Clipchamp.mp4', help='Input source (file/folder)')
    parser.add_argument('--img-size', type=int, default=640, help='Inference size (pixels)')
    parser.add_argument('--device', default='0,1,2,3', help='Device: cpu or cuda')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='Directory to save results')
    opt = parser.parse_args()

    with torch.no_grad():
        import threading
        detect_thread = threading.Thread(target=detect, args=(cfg, opt))
        detect_thread.start()


        # Optionally, join threads if needed
        detect_thread.join()
