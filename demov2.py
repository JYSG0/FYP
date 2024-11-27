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
from scipy.ndimage import label

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

from scipy.spatial.distance import cdist
import math

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
class YOLOP:

    def start_recording(self, img_det, save_path, fps, fourcc='mp4v'):
        """
        Start video recording.
        """
        self.vid_writer = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            (img_det.shape[1], img_det.shape[0])
        )
        self.is_recording = True
        print(f"Recording started. Saving to {save_path}")

    def stop_recording(self):
        """
        Stop video recording.
        """
        if self.vid_writer:
            self.vid_writer.release()
            self.vid_writer = None
            self.is_recording = False
            print("Recording stopped and video saved.")

    def __init__(self, cfg, opt):
        self.cfg = cfg
        self.opt = opt
        self.is_recording = False
        self.vid_writer = None


    def detect(self):
        cfg = self.cfg
        opt = self.opt
        y = 600
        point = False
        logger, _, _ = create_logger(
            cfg, cfg.LOG_DIR, 'demo')

        device = select_device(logger,opt.device)
        if os.path.exists(opt.save_dir):  # output dir
            shutil.rmtree(opt.save_dir)  # delete dir
        os.makedirs(opt.save_dir)  # make new dir
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = get_net(cfg)
        checkpoint = torch.load(opt.weights, map_location= device)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        if half:
            model.half()  # to FP16

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
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


        # Run inference
        t0 = time.time()

        vid_path, vid_writer = None, None
        img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        model.eval()

        inf_time = AverageMeter()
        nms_time = AverageMeter()
        
        for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
            img = transform(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            t1 = time_synchronized()
            det_out, da_seg_out,ll_seg_out= model(img)
            t2 = time_synchronized()
            # if i == 0:
            #     print(det_out)
            inf_out, _ = det_out
            inf_time.update(t2-t1,img.size(0))

            # Apply NMS
            t3 = time_synchronized()
            det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
            t4 = time_synchronized()

            nms_time.update(t4-t3,img.size(0))
            det=det_pred[0]

            save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")

            _, _, height, width = img.shape
            h,w,_=img_det.shape
            pad_w, pad_h = shapes[1][1]
            pad_w = int(pad_w)
            pad_h = int(pad_h)
            ratio = shapes[1][0][1]

            da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
            da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
            _, da_seg_mask = torch.max(da_seg_mask, 1)
            da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
            # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)

            
            ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
            ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
            _, ll_seg_mask = torch.max(ll_seg_mask, 1)
            ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
            # Lane line post-processing
            #ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
            #ll_seg_mask = connect_lane(ll_seg_mask)

            img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

            # Draw the red dot
            dot_color = (255, 0, 255)  # BGR for red
            dot_radius = 5  # Radius of the dot
            cv2.circle(img_det, (640, 715), dot_radius, dot_color, -1)  # -1 to fill the circle

            # Extract lane pixels from the mask
            lane_pixels = np.argwhere(ll_seg_mask > 0)
                        
            # Original resolution
            old_width, old_height = 1920, 2440

            # New resolution
            new_width, new_height = 1280, 720

            # Scale factors
            width_scale = new_width / old_width
            height_scale = new_height / old_height

            # Lane pixels at original resolution
            lane_pixels = np.argwhere(ll_seg_mask > 0)

            # Scale the coordinates
            scaled_lane_pixels = np.round(lane_pixels * [height_scale, width_scale]).astype(int)

            # Group by unique y-values
            lanes_by_y = {}
            for y, x in scaled_lane_pixels:
                if y not in lanes_by_y:
                    lanes_by_y[y] = []
                lanes_by_y[y].append(x)

            # Determine the two nearest x-values for each y-value
            for y, x_values in lanes_by_y.items():
                print(y)
                if y == 600:

                    x_values = lanes_by_y.get(y, [])
                   

                    # if not x_values:
                    #     y -= 5
                    #     continue

                    x_values = np.array(x_values)
                    # Sort x-values by their distance to middle_x
                    sorted_x = sorted(x_values, key=lambda x: abs(x - 640))

                    # Initialize x1 and x2
                    x1, x2 = None, None

                    # Find one x-value below the midpoint and one above
                    for x in sorted_x:
                        if x < 640 and x1 is None:
                            x1 = x
                        elif x > 640 and x2 is None:
                            x2 = x
                        if x1 is not None and x2 is not None:
                            break


                    if x1 is not None and x2 is not None:
                        # Calculate the midpoint
                        midpoint = int((x1 + x2) / 2)
                        # Draw a dot at the midpoint
                        dot_color1 = (255, 255, 255)  # BGR for magenta
                        dot_radius1 = 5  # Radius of the dot                       

                        cv2.circle(img_det, (midpoint, y), dot_radius1, dot_color1, -1)  # -1 to fill the circle
                    

                        # Calculate the distance using the Euclidean formula
                        distance = math.sqrt((midpoint - 640)**2 + (y - 715)**2)
                        # Draw the line between the points
                        cv2.line(img_det, (640, 715), (midpoint, y), (0, 255, 0), 2)  # Green line
                        # Output the distance
                        print(f"Distance between the parallel lines: {distance:.2f} units")
                        # Add distance text
                        distance_text = f"{distance:.2f} units"
                        midpoint_text_x = int((640 + midpoint) / 2)
                        midpoint_text_y = int((715 + y) / 2)
                        point = True
                        cv2.putText(img_det, distance_text, (midpoint_text_x + 10, midpoint_text_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        
                    # Proceed if valid x1 and x2 are found
                    elif (x1 is None or x2 is None) and point:
                        cv2.circle(img_det, (midpoint, y), dot_radius1, dot_color1, -1)  # -1 to fill the circle
                        # Debug: Print the detected lanes and midpoint
                        print(f"At y={y}, closest x values: {x1}, {x2}. Midpoint: {midpoint}")
                        # Calculate the distance using the Euclidean formula
                        distance = math.sqrt((midpoint - 640)**2 + (y - 715)**2)
                        # Draw the line between the points
                        cv2.line(img_det, (640, 715), (midpoint, y), (0, 255, 0), 2)  # Green line
                        # Output the distance
                        print(f"Distance between the parallel lines: {distance:.2f} units")
                        # Add distance text
                        distance_text = f"{distance:.2f} units"
                        midpoint_text_x = int((640 + midpoint) / 2)
                        midpoint_text_y = int((715 + y) / 2)
                        cv2.putText(img_det, distance_text, (midpoint_text_x + 10, midpoint_text_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        break 
                                        
            # If no midpoint was found after trying all y-values
            if y < 500:
                print("No valid midpoint found in the specified range.")

            if len(det):
                det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
                for *xyxy,conf,cls in reversed(det):
                    label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)


            if dataset.mode == 'images':
                cv2.imwrite(save_path,img_det)

            elif dataset.mode == 'video':
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    h,w,_=img_det.shape
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(img_det)
                cv2.imshow('image', img_det)
                cv2.waitKey(1)

            else:
                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    if not self.is_recording:
                        self.start_recording(img_det, save_path, fps)
                    
                    if self.is_recording:
                        self.vid_writer.write(img_det)

                    cv2.imshow('image', img_det)
                    cv2.waitKey(1)

        print('Results saved to %s' % Path(opt.save_dir))
        print('Done. (%.3fs)' % (time.time() - t0))
        print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))




    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
        parser.add_argument('--source', type=str, default='6', help='source')  # file/folder   ex:inference/images
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        opt = parser.parse_args()
        with torch.no_grad():
            detect(cfg,opt)
