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

from ultralytics import YOLO
from collections import Counter
from realsense_depth import *
from ultralytics.utils.plotting import Annotator, colors

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

# Initialize Pygame for PS4 controller vibration
pygame.init()
pygame.joystick.init()

# Check for connected joysticks
if pygame.joystick.get_count() == 0:
    print("No joystick connected!")
    exit()

# Get the first joystick
joystick = pygame.joystick.Joystick(0)
joystick.init()

# global cap for camera
cap = DepthCamera()

# Load the model
modely = YOLO("best.pt")
namesy = modely.names

#object detection class_id
class_ids=[]

run_detection = True

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

joysticks = {}
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    joysticks[joystick.get_instance_id()] = joystick
else:
    print("No PS4 controller detected. Vibration will not be available.")
        
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

#gets centre of bounding bvox
def get_center_of_bbox(box):
    """ Calculate the center point of the bounding box """
    x_center = int((box[0] + box[2]) / 2)
    y_center = int((box[1] + box[3]) / 2)
    return (x_center, y_center)
    
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
    h, w = da_seg_mask.shape
    vehicle_center = w // 2

    # Detect drivable area
    front_row = da_seg_mask[int(h * 0.8):, :]  # Bottom 20% of the frame
    drivable = np.sum(front_row) > 0

    # Analyze lane position
    bottom_row = ll_seg_mask[-1, :]  # Bottom row of the lane mask
    lane_pixels = np.where(bottom_row > 0)[0]

    if len(lane_pixels) >= 2:
        left_lane = lane_pixels[0]
        right_lane = lane_pixels[-1]
        lane_center = (left_lane + right_lane) // 2

        if vehicle_center < left_lane:
            lane_position = "left"
            cv2.putText(img_det, "Lane Shift Left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        elif vehicle_center > right_lane:
            lane_position = "right"
            cv2.putText(img_det, "Lane Shift Right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        else:
            lane_position = "center"
            cv2.putText(img_det, "Good Lane Keeping", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        lane_position = "unknown"

    return drivable, lane_position

def object_detection(class_ids, odrv0):
        global detected
        # Control motors based on object detections

        # People or stop sign detected
        if 4 in class_ids or 8 in class_ids:
            print("Stop")
            # run_detection = False
            Auto_driving = False
            esp = False
            detected = True
            odrv0.axis1.controller.input_vel = 0
            odrv0.axis0.controller.input_vel = 0

        # Slow or Speed sign detected
        elif 6 in class_ids or 7 in class_ids:
            detected = True
            print("Slow")
            odrv0.axis1.controller.input_vel = -0.8
            odrv0.axis0.controller.input_vel = -0.8

        # Hump or Pedestrain Sign detected
        if 1 in class_ids or 3 in class_ids:
            detected = True
            print("Pedestrain")
            odrv0.axis1.controller.input_vel = -0.8
            odrv0.axis0.controller.input_vel = -0.8
        
        # Hump or pedestrain sign AND people detected
        if  (1 in class_ids or 3 in class_ids) and 4 in class_ids:
            detected = True
            print("Stop")
            odrv0.axis1.controller.input_vel = 0
            odrv0.axis0.controller.input_vel = 0

        if not class_ids:
            detected = False
# Main detection and control loop
def detect(cfg, opt):
    #global variables to pass
    global class_ids, annotated_frame, run_detection, detected, is_recording, vid_counter, img_counter, img_counter1
    is_recording = False
    detected = False
    run_detection = True
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'demo')
    device = select_device(logger, opt.device)

    # Prepare output directory
    if os.path.exists(opt.save_dir):
        shutil.rmtree(opt.save_dir)
    os.makedirs(opt.save_dir)

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

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

        # Get Camera
        ret, success, frame = cap.get_frame()
        idx = 0

        if not ret:
            print("Failed to grab frame")
            return

        # Run YOLO detection if enabled
        if run_detection:
            results = modely(frame, conf=0.5, verbose=False, device=0)  # Running detection
            annotated_frame = results[0].plot()  # Annotate frame with segmentaion mask
            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(annotated_frame, line_width=2, example=namesy)

            if boxes is not None:
                for box, cls in zip(boxes, clss):
                    idx += 1
                    annotator.box_label(box, color=colors(int(cls), True), label=namesy[int(cls)])

                    crop_obj = annotated_frame[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
                    center_point = get_center_of_bbox(box)

                    cv2.circle(annotated_frame, center_point, 5, (0, 0, 255), -1)
                    distance = success[center_point[1], center_point[0]]
                    cv2.putText(annotated_frame, "{}mm".format(distance), (center_point[0], center_point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

            # Process and print class names for each detection
            for result in results:
                class_ids = result.boxes.cls.tolist()
                class_names = [modely.names[int(id)] for id in class_ids]  # Map IDs to names
            
            # Count occurrences of each class name
            class_counts = Counter(class_names)
            for class_name, count in class_counts.items():
                print(f"{count} {class_name} detected.")
        else:
            results = modely(frame, conf=0.5, verbose=False)  # Running detection
            annotated_frame = results[0].plot()

        # Analyze results
        drivable, lane_position = process_segmentation(da_seg_mask, ll_seg_mask, img_det)
    
        # Control logic
        object_detection(class_ids, odrv0)
        if not detected:
            if drivable:
                print("Drivable area detected.")
                odrv0.axis0.controller.input_vel = -1.0  # Move forward
                odrv0.axis1.controller.input_vel = -1.0  # Move forward
                cv2.putText(img_det, "Drivable Area", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                print("No drivable area detected. Stopping.")
                odrv0.axis0.controller.input_vel = 0  # Stop
                odrv0.axis1.controller.input_vel = 0  # Stop

            if lane_position == "left" and not detected:
                print("Vehicle drifting left. Slowing down.")
                odrv0.axis0.controller.input_vel = -0.8
                odrv0.axis1.controller.input_vel = -1   # Stop

            elif lane_position == "right" and not detected:
                print("Vehicle drifting right. Slowing down.")
                odrv0.axis0.controller.input_vel = -1
                odrv0.axis1.controller.input_vel = -0.8   # Stop

            elif lane_position == "center":
                print("Vehicle centered in lane.")

                        # GPU memory usage
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # In MB
        reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 2)    # In MB
        free_memory = reserved_memory - allocated_memory                      # Free within reserved

        # print(f"Allocated Memory: {allocated_memory:.2f} MB")
        # print(f"Reserved Memory: {reserved_memory:.2f} MB")
        # print(f"Free Memory within Reserved: {free_memory:.2f} MB")
        cv2.imshow('YOLOP Inference', img_det)
        # Display the annotated frame
        cv2.imshow('YOLOv8 Detection', annotated_frame)

        if is_recording:
            out.write(annotated_frame)
            out1.write(img_det)
        key = cv2.waitKey(1) & 0xFF  # Adjust delay based on video FPS

        # 'r' key to start/stop video recording
        if key == ord('r'):
            if not is_recording:
                print("Recording started...")
                vid_name = os.path.join(output_video_dir, f"output_video_{vid_counter}.mp4")  # Set video name
                vid_name1 = os.path.join(output_video_dir, f"lane_video_{vid_counter}.mp4")  # Set video name
               
                # Define the codec and create a VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(vid_name, fourcc, 20.0, (annotated_frame.shape[1], annotated_frame.shape[0]))
                out1 = cv2.VideoWriter(vid_name1, fourcc, 20.0, (img_det.shape[1], img_det.shape[0]))
                is_recording = True
            else:
                print("Recording stopped.")
                is_recording = False
                print(f"{vid_name} written!")
                print(f"{vid_name1} written!")

                vid_counter += 1
                out.release()  # Stop recording and release the output file
                out1.release()

        # 's' key to capture an image
        elif key == ord('s'):
            img_name = os.path.join(output_image_dir, f"opencv_frame_{img_counter}.png")  # Set image name
            img_name1 = os.path.join(output_image_dir, f"lanes_frame_{img_counter}.png")  # Set image name

            cv2.imwrite(img_name, annotated_frame)  # Write image file
            cv2.imwrite(img_name1, img_det)  # Write image file

            print(f"{img_name} written!")
            print(f"{img_name1} written!")

            img_counter += 1

        # Stop on key press
        if key == ord('q'):
            emergency_stop(odrv0)
            break

    emergency_stop(odrv0)
    print("Inference completed.")
    pygame.quit
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/End-to-end.pth', help='Path to model weights')
    parser.add_argument('--source', type=str, default='6', help='Input source (file/folder)')
    parser.add_argument('--img-size', type=int, default=640, help='Inference size (pixels)')
    parser.add_argument('--device', default='0,1,2,3', help='Device: cpu or cuda')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='Directory to save results')
    opt = parser.parse_args()

    with torch.no_grad():
        detect(cfg, opt)
