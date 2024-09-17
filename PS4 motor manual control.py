import argparse
import csv
import os
import platform
import sys
import time
import datetime
from pathlib import Path
import threading
import torch
import pygame
import Jetson.GPIO as GPIO
import odrive
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (
    LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2, increment_path,
    non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh
)
from utils.torch_utils import select_device, smart_inference_mode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# ODrive and motor control setup
emergency_stop_flag = False
stop_sign_detected = False
stop_sign_timestamp = None
stop_sign_handled = False
cooldown_timestamp = None
turn_right_detected = False
manual_override = False

valid_pins = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

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

# Function to stop the motor and exit the program
def emergency_stop():
    global emergency_stop_flag
    emergency_stop_flag = True
    try:
        odrv0.axis0.controller.input_pos = 0
        GPIO.output(pwm, GPIO.HIGH)
        GPIO.output(steering, GPIO.HIGH)
        odrv0.axis0.requested_state = 1  # Set ODrive to idle state
    except Exception as e:
        print("Error during emergency stop:", e)
    finally:
        print("Emergency Stop!")

# Function to handle trigger control
def handle_triggers():
    # Left trigger (L2) will decrease the motor position
    if joystick.get_axis(left_trigger_axis) > 0.1:  # Small threshold to avoid accidental movements
        odrv0.axis0.controller.input_pos -= position_increment
        print(f"Moving backward, position: {odrv0.axis0.controller.input_pos}")
    # Right trigger (R2) will increase the motor position
    if joystick.get_axis(right_trigger_axis) > 0.1:  # Small threshold to avoid accidental movements
        odrv0.axis0.controller.input_pos += position_increment
        print(f"Moving forward, position: {odrv0.axis0.controller.input_pos}")

# YOLOv5 Inference Function (remains unchanged)
@smart_inference_mode()
def run(stop_event, weights=ROOT / "yolov5s.pt", source=ROOT / "data/images", data=ROOT / "data/coco128.yaml", imgsz=(640, 640), 
        conf_thres=0.25, iou_thres=0.45, max_det=1000, device="", view_img=False, save_txt=False, save_csv=False, 
        save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, 
        update=False, project=ROOT / "runs/detect", name="exp", exist_ok=False, line_thickness=3, hide_labels=False, 
        hide_conf=False, half=False, dnn=False, vid_stride=1):
    """Runs YOLOv5 detection inference on various sources like images, videos, directories, streams, etc."""
    
    global stop_sign_detected, stop_sign_timestamp, stop_sign_handled, cooldown_timestamp, turn_right_detected, manual_override
    # Load model and perform inference (remains unchanged)
    # ...

def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    # Parse arguments (remains unchanged)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main_loop(opt):
    stop_event = threading.Event()
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    yolo_thread = threading.Thread(target=run, args=(stop_event,), kwargs=vars(opt))
    yolo_thread.start()

    global emergency_stop_flag, stop_sign_detected, stop_sign_timestamp, stop_sign_handled, cooldown_timestamp, manual_override, turn_right_detected
    running = True
    while running and not emergency_stop_flag:
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN and event.button == 1:  # Circle button on PS4 controller
                manual_override = not manual_override  # Toggle manual override
                print("Manual override:", "Activated" if manual_override else "Deactivated")
                
                if manual_override:
                    stop_sign_detected = False
                    stop_sign_handled = False
                    odrv0.axis0.requested_state = 8  # Reactivate ODrive
                    GPIO.output(pwm, GPIO.LOW)
                    GPIO.output(steering, GPIO.LOW)

        # Use the triggers to control motor position
        handle_triggers()

        # Logic for stop sign detection, turning, and joystick inputs (removed joystick handling)
        # ...

        # Emergency stop with B Button
        if joystick.get_button(2) == 1:  # B Button
            stop_event.set()  # Signal the YOLOv5 thread to stop
            yolo_thread.join()  # Wait for the YOLOv5 thread to finish
            emergency_stop()       

        time.sleep(0.1)
        
    yolo_thread.join()
    print("Yolov5 thread has ended.")    

# Connect to ODrive
odrv0 = connect_to_odrive()

# Display drive voltage
print("ODrive Voltage:", odrv0.vbus_voltage)

# Initialize GPIOs
pwm = 12
steering = 13
limit2 = 26  # Left limit switch
limit1 = 23  # Right limit switch
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup([pwm, steering], GPIO.OUT, initial=GPIO.LOW)
GPIO.setup([limit1, limit2], GPIO.IN)

# Initialize Pygame and the PS4 controller
pygame.display.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

# Set up motor control parameters
position_increment = 0.05  # Adjust this to control how much the position changes with each trigger press
left_trigger_axis = 4  # L2 trigger axis
right_trigger_axis = 5  # R2 trigger axis

# Activate the ODrive
odrv0.axis0.requested_state = 8

# Run the main loop
if __name__ == "__main__":
    opt = parse_opt()
    main_loop(opt)
