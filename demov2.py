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
            # ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
            # ll_seg_mask = connect_lane(ll_seg_mask)

            img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

            # Extract lane pixels from the mask
            lane_pixels = np.argwhere(ll_seg_mask > 0)

            # Extract the x-values of the lane pixels
            x_values = lane_pixels[:, 1]  # Extract x-coordinates
            y_values = lane_pixels[:, 0]  # Extract y-coordinates (optional if needed)

            # Sort lane_pixels based on y-values to ensure proper order
            sorted_indices = np.argsort(y_values)
            x_values = x_values[sorted_indices]
            y_values = y_values[sorted_indices]

            # Compute the differences between consecutive x-values
            x_differences = np.abs(np.diff(x_values))

            # Define a threshold for detecting sudden changes in x-values
            threshold = 100  # Adjust this threshold as needed

            # Identify where the differences exceed the threshold
            lane_change_indices = np.where(x_differences > threshold)[0]

            # Check if multiple lanes are detected
            if len(lane_change_indices) > 0:
                print("Multiple lanes detected!")
                for pixel in lane_pixels:
                    y, x = pixel  # Row (y) and column (x) of the pixel
                    print(f"Lane pixel at (x={x}, y={y})")
                    # Convert to a numpy array for easier processing
                    lane_pixels = np.array(lane_pixels)

                    # # Sort by the x-coordinate
                    # lane_pixels = lane_pixels[lane_pixels[:, 1].argsort()]

                    # # Threshold to determine whether to start a new group
                    # x_threshold = 60

                    # # Separate into clusters
                    # clusters = []
                    # current_cluster = [lane_pixels[0]]  # Start with the first pixel

                    # for i in range(1, len(lane_pixels)):
                    #     if abs(lane_pixels[i][1] - lane_pixels[i - 1][1]) > x_threshold:
                    #         # Start a new cluster
                    #         clusters.append(current_cluster)
                    #         current_cluster = [lane_pixels[i]]
                    #     else:
                    #         # Add to the current cluster
                    #         current_cluster.append(lane_pixels[i])

                    # # Add the last cluster
                    # if current_cluster:
                    #     clusters.append(current_cluster)

                    # # Convert each cluster back to a list of lists
                    # clusters = [np.array(cluster).tolist() for cluster in clusters]
                    import matplotlib.pyplot as plt
                    from sklearn.cluster import DBSCAN

                    # DBSCAN clustering
                    db = DBSCAN(eps=60, min_samples=5).fit(lane_pixels)
                    labels = db.labels_
                    print("number of clusters:",len(set(labels)))
                    plt.figure()
                    plt.scatter(lane_pixels[:,0],lane_pixels[:,1], c= labels)
                    plt.show()
                    # # Organize points into clusters
                    # clusters = []
                    # unique_labels = set(labels)
                    # for label in unique_labels:
                    #     if label != -1:  # Exclude noise points
                    #         # Find indices where the label matches
                    #         cluster_indices = np.where(labels == label)[0]
                    #         clusters[label] = lane_pixels[cluster_indices].tolist()
                    #     print(f"DBSCAN Clusters: {clusters}")

                    # import csv

                    # # Save to CSV
                    # csv_file = "dynamic_lane_clusters.csv"
                    # with open(csv_file, "w", newline="") as f:
                    #     writer = csv.writer(f)
                        
                    #     # Write headers
                    #     headers = [f"Cluster {i+1}" for i in range(len(clusters))]
                    #     writer.writerow(headers)
                        
                    #     # Find the longest cluster
                    #     max_length = max(len(points) for points in clusters.values())
                        
                    #     # Write rows
                    #     for i in range(max_length):
                    #         row = []
                    #         for cluster_points in clusters.values():
                    #             row.append(cluster_points[i] if i < len(cluster_points) else "")
                    #         writer.writerow(row)

                    # print(f"Clusters saved to {csv_file}")

                    # # Print the clusters
                    # for idx, cluster in enumerate(clusters):
                    #     print(f"Cluster {idx + 1}:")
                    #     for coord in cluster:
                            # print(f"[x: {coord[1]}, y: {coord[0]}]")  # Format x and y
                    # # Create a blank image
                    # img_height, img_width = 1500, 1500
                    # img1 = np.zeros((img_height, img_width, 3), dtype=np.uint8)

                    # # Colors for clusters
                    # colors = [(0, 255, 0), (0, 0, 255)]  # Green, Red

                    # # Iterate through clusters and draw lines
                    # for cluster_idx, cluster in enumerate(clusters):
                    #     color = colors[cluster_idx % len(colors)]  # Cycle through colors if more clusters
                    #     for i in range(len(cluster) - 1):  # Loop through each consecutive point
                    #         pt1 = (cluster[i][1], cluster[i][0])   # (x, y)
                    #         pt2 = (cluster[i + 1][1], cluster[i + 1][0])  # (x, y)
                    #         cv2.line(img1, pt1, pt2, color, thickness=2)  # Draw line between consecutive points
                    #     # Show the image
                    #     cv2.imshow('Lines from Clusters', img1)
                    #     cv2.waitKey(1)
            else:
                print("No multiple lanes detected.")

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