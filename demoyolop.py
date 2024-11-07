import time
from pathlib import Path
import cv2
import torch

# Import necessary utilities
from utils import (
    time_synchronized, select_device, increment_path,
    scale_coords, xyxy2xywh, non_max_suppression,
    split_for_trace_model, driving_area_mask,
    lane_line_mask, plot_one_box, show_seg_result,
    AverageMeter, LoadImages
)

def detect():
    # Specify paths to model and video directly
    weights = 'yolopv2.pt'  # Update this path
    source = 'SingaporeRoad.mp4'  # Update this path
    img_size = 640
    conf_thres = 0.3
    iou_thres = 0.45
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_img = True  # Save inference images

    # Set up directories and device
    save_dir = Path(increment_path(Path('runs/detect') / 'exp', exist_ok=True))  # Increment save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # Load model
    stride = 32
    model = torch.jit.load(weights, map_location=device)  # Load TorchScript model
    model.to(device).eval()

    if device == 'cuda':
        model.half()  # Use FP16 for faster inference

    # Set up the video source
    dataset = LoadImages(source, img_size=img_size, stride=stride)
    vid_path, vid_writer = None, None

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if device == 'cuda' else img.float()
        img /= 255.0  # Normalize to 0 - 1

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img)
        t2 = time_synchronized()

        # Process outputs
        pred = split_for_trace_model(pred, anchor_grid)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # Loop through detections
        for i, det in enumerate(pred):
            im0 = im0s.copy()

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    plot_one_box(xyxy, im0, line_thickness=3)

            # Overlay segmentation masks
            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)

            # Save results
            if save_img:
                save_path = str(save_dir / Path(path).name)
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # Video mode
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    inf_time.update(t2 - t1, img.size(0))
    nms_time.update(t4 - t3, img.size(0))
    waste_time.update(tw2 - tw1, img.size(0))
    print(f'Done. ({time.time() - t0:.3f}s)')

    if vid_writer:
        vid_writer.release()

if __name__ == '__main__':
    detect()
