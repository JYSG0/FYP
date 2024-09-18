#Stop_Sign can be detected from far away
#Turn Right must be within 80cm from the camera

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

# Function to handle D-Pad control
def handle_dpad(event):
    if event.value == (0, 1):  # UP
        odrv0.axis0.controller.input_pos += 0.1
    elif event.value == (0, -1):  # DOWN
        odrv0.axis0.controller.input_pos -= 0.1
       
    elif event.value == (1, 0):  # LEFT
        odrv0.axis0.requested_state = 1  # Set ODrive to idle state      	
        odrv0.axis1.requested_state = 1  # Set ODrive to idle state      
    elif event.value == (-1, 0):  # RIGHT
        odrv0.axis0.requested_state = 8  # Set ODrive to closed loop state     
        odrv0.axis1.requested_state = 8  # Set ODrive to closed loop state         	
# Function to handle joystick control
def handle_joystick(move_left, move_right):
    if joystick.get_axis(1) <= -0.5:  # Joystick UP
        odrv0.axis1.controller.input_pos += 0.1        #right
    elif joystick.get_axis(1) >= 0.5:  # Joystick DOWN
        odrv0.axis1.controller.input_pos -= 0.1 	 #right
    if joystick.get_axis(4) <= -0.5:  # Joystick UP
        odrv0.axis0.controller.input_pos += 0.1        #left
    elif joystick.get_axis(4) >= 0.5:  # Joystick DOWN
        odrv0.axis0.controller.input_pos -= 0.1 	 #left
        
    if joystick.get_axis(0) <= -0.5 and move_left:  # Joystick LEFT
        GPIO.output(pwm, GPIO.LOW)
        GPIO.output(steering, GPIO.HIGH)
    elif joystick.get_axis(0) >= 0.5 and move_right:  # Joystick RIGHT
        GPIO.output(pwm, GPIO.LOW)
        GPIO.output(steering, GPIO.LOW)
    else:  # Joystick IDLE
        GPIO.output(pwm, GPIO.HIGH)
        GPIO.output(steering, GPIO.HIGH)

# Function to handle limit switch detection
def check_limit_switches():
    global emergency_stop_flag
    while not emergency_stop_flag:
        if GPIO.input(limit1) == GPIO.HIGH and GPIO.input(limit2) == GPIO.HIGH:  # Right Limit Switch
            print("Right")
            return False, True
        elif GPIO.input(limit1) == GPIO.LOW and GPIO.input(limit2) == GPIO.HIGH:  # Left Limit Switch
            print("Left")
            return True, False
        else:
            print("Within Movable Range")
            return True, True

# YOLOv5 Inference Function
@smart_inference_mode()
def run(stop_event, weights=ROOT / "yolov5s.pt", source=ROOT / "data/images", data=ROOT / "data/coco128.yaml", imgsz=(640, 640), 
        conf_thres=0.25, iou_thres=0.45, max_det=1000, device="", view_img=False, save_txt=False, save_csv=False, 
        save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, 
        update=False, project=ROOT / "runs/detect", name="exp", exist_ok=False, line_thickness=3, hide_labels=False, 
        hide_conf=False, half=False, dnn=False, vid_stride=1):
    """Runs YOLOv5 detection inference on various sources like images, videos, directories, streams, etc."""
    
    global stop_sign_detected, stop_sign_timestamp, stop_sign_handled, cooldown_timestamp, turn_right_detected, manual_override
    
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)
        
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Dataloader
    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None]*bs, [None]*bs
    
    #run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        if stop_event.is_set():
            break
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None] # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        #inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"
        
        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            save_path = str(save_dir / p.name) # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")
            s += "%gx%g " % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] # normalization gain whwh
            imc = im0.copy() if save_crop else im0 # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum() # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, " # add to string


                # Write results and check for stop sign
                stop_sign_in_frame = False
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c] if hide_conf else f"{names[c]}"
                    current_time = time.time()
                    if label == "Stop_Sign":
                    	stop_sign_in_frame = True
                    	#current_time = time.time()
                    	if not stop_sign_handled:
                    		if (cooldown_timestamp is None or current_time - cooldown_timestamp >= 5):
                    			stop_sign_detected = True
                    			stop_sign_timestamp = current_time
                    			stop_sign_handled = True
                    			cooldown_timestamp = current_time
                    elif label == "Turn right ahead":
                    	turn_right_detected = True
                    	print("Turn right ahead sign detected, turning vehicle right.")
                    else:
                    	turn_right_detected = False
                    
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img: # Add bbox to image
                        c = int(cls)
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)
                
                # To reset stop sign handled flag if there is no stop sign in frame
                if not stop_sign_in_frame:
                	stop_sign_handled = False

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)
        # Print time (inference-only)
        #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)
    #LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])

def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.50, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main_loop(opt):
    stop_event = threading.Event()
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    #threading.Thread(target=run, kwargs=vars(opt)).start()
    yolo_thread = threading.Thread(target=run, args=(stop_event,), kwargs=vars(opt))
    yolo_thread.start()

    global emergency_stop_flag, stop_sign_detected, stop_sign_timestamp, stop_sign_handled, cooldown_timestamp, manual_override, turn_right_detected
    running = True
    move_left = True
    move_right = True
    while running and not emergency_stop_flag:
        for event in pygame.event.get():
            if event.type == pygame.JOYHATMOTION:
                handle_dpad(event)
            if event.type == pygame.JOYBUTTONDOWN and event.button == 1:  # Circle button on PS4 controller
                manual_override = not manual_override  # Toggle manual override
                print("Manual override:", "Activated" if manual_override else "Deactivated")
                
                if manual_override:
                    # Immediately cancel the stop action and reactivate the vehicle
                    stop_sign_detected = False
                    stop_sign_handled = False
                    odrv0.axis0.requested_state = 8  # Reactivate ODrive
                    odrv0.axis1.requested_state = 8 # Reactivate ODrive
                    GPIO.output(pwm, GPIO.LOW)
                    GPIO.output(steering, GPIO.LOW)


        move_left, move_right = check_limit_switches()
        handle_joystick(move_left, move_right)
        
        # Check for stop sign detection
        if stop_sign_detected and not manual_override:
            current_time = time.time()
            if current_time - stop_sign_timestamp < 5:
                print("Stop sign detected. Stopping for 5 seconds.")
                odrv0.axis0.controller.input_pos = 0
                GPIO.output(pwm, GPIO.HIGH)
                GPIO.output(steering, GPIO.HIGH)
                odrv0.axis0.requested_state = 1  # Set ODrive to idle state
            else:
                stop_sign_detected = False
                stop_sign_handled = False  # Reset to allow detection again
                odrv0.axis0.requested_state = 8  # Reactivate ODrive
        
        #duplicate for left turn        
        if turn_right_detected and GPIO.input(limit1) == GPIO.LOW and GPIO.input(limit2) == GPIO.HIGH:
        	#Limit switch will stop AI right turn
        	print("Max Turning vehicle right.")
        	GPIO.output(pwm, GPIO.HIGH)
        	GPIO.output(steering, GPIO.HIGH)
        elif turn_right_detected:
        	#move_left, move_right = check_limit_switches()
        	print("Turning vehicle right.")
        	GPIO.output(pwm, GPIO.LOW)
        	GPIO.output(steering, GPIO.LOW)

        

        print("Wheel Turns:", "%.1f" % odrv0.axis0.controller.input_pos)
        print("GPIO 12 (PWM) State:", GPIO.input(12))  # Print state of GPIO 12
        print("GPIO 13 (Steering) State:", GPIO.input(13))  # Print state of GPIO 13
        print("GPIO 26 State:", GPIO.input(26))  # Print state of GPIO 26
        print("GPIO 23 State:", GPIO.input(23))  # Print state of GPIO 23

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

# Initialize the display subsystem of Pygame
pygame.display.init()

# Initialize the joystick module
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

# Flag to indicate emergency stop
emergency_stop_flag = False

# Activate the ODrive
odrv0.axis0.requested_state = 8
odrv0.axis1.requested_state = 8 

# Run the main loop
if __name__ == "__main__":
    opt = parse_opt()
    main_loop(opt)
