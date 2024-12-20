
import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio

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

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


# Function to find and connect to ODrive
def connect_to_odrive():
    try:
        odrv = odrive.find_any()
        if odrv is None:
            print("ODrive not found. Exiting program.")
            sys.exit()
        return odrv
    except Exception as e:
        print("Error connecting to ODrive:", e)
        sys.exit()


# Function to stop the motor (emergency stop)
def emergency_stop(odrv0):
    try:
        odrv0.axis0.controller.input_vel = 0
        print("Emergency Stop Activated!")
    except Exception as e:
        print("Error during emergency stop:", e)


# Function to process drivable area and lane lines
def process_segmentation(da_seg_mask, ll_seg_mask):
    h, w = da_seg_mask.shape
    vehicle_center = w // 2

    # Check if there is a drivable area in front of the vehicle
    front_row = da_seg_mask[int(h * 0.8):, :]  # Bottom 20% of the image
    if np.sum(front_row) > 0:  # If green area exists
        drivable = True
    else:
        drivable = False

    # Check lane alignment
    bottom_row = ll_seg_mask[-1, :]  # Bottom row of the lane mask
    lane_pixels = np.where(bottom_row > 0)[0]
    if len(lane_pixels) >= 2:
        left_lane = lane_pixels[0]
        right_lane = lane_pixels[-1]
        lane_center = (left_lane + right_lane) // 2

        if vehicle_center < left_lane:
            lane_position = "left"
        elif vehicle_center > right_lane:
            lane_position = "right"
        else:
            lane_position = "center"
    else:
        lane_position = "unknown"

    return drivable, lane_position


# Main inference and control loop
def detect(cfg, opt):
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'demo')
    device = select_device(logger, opt.device)

    if os.path.exists(opt.save_dir):  # Output directory
        shutil.rmtree(opt.save_dir)  # Delete existing directory
    os.makedirs(opt.save_dir)  # Create new directory

    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device).eval()

    dataset = LoadImages(opt.source, img_size=opt.img_size)
    odrv0 = connect_to_odrive()  # Connect to ODrive
    odrv0.axis0.requested_state = 8  # Set ODrive to closed-loop velocity control mode

    for path, img, img_det, vid_cap, shapes in tqdm(dataset, total=len(dataset)):
        img = transform(img).to(device)
        img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        _, da_seg_out, ll_seg_out = model(img)
        t2 = time_synchronized()

        # Post-process segmentation results
        _, _, height, width = img.shape
        da_seg_mask = torch.nn.functional.interpolate(da_seg_out, scale_factor=1, mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.squeeze().cpu().numpy()

        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_out, scale_factor=1, mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.squeeze().cpu().numpy()

        # Analyze drivable area and lane position
        drivable, lane_position = process_segmentation(da_seg_mask, ll_seg_mask)

        if drivable:
            print("Drivable area detected.")
            odrv0.axis0.controller.input_vel = 1.0  # Set forward velocity
        else:
            print("No drivable area detected. Stopping.")
            odrv0.axis0.controller.input_vel = 0  # Stop

        if lane_position == "left":
            print("Vehicle is drifting left. Adjusting.")
            odrv0.axis0.controller.input_vel = 0.8  # Slow down slightly
        elif lane_position == "right":
            print("Vehicle is drifting right. Adjusting.")
            odrv0.axis0.controller.input_vel = 0.8  # Slow down slightly
        elif lane_position == "center":
            print("Vehicle is well-centered in the lane.")

        # Show or save result
        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)
        cv2.imshow("YOLOP Inference", img_det)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            odrv0.axis0.requested_state = 1

            break

    emergency_stop(odrv0)
    print("Inference completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='file/folder')  # File/folder input
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    opt = parser.parse_args()

    with torch.no_grad():
        detect(cfg, opt)
