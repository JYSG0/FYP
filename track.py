import os

import cv2
import pyrealsense2
from realsense_depth import *
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

model = YOLO("best.pt")
names = model.names

def get_center_of_bbox(box):
    """ Calculate the center point of the bounding box """
    x_center = int((box[0] + box[2]) / 2)
    y_center = int((box[1] + box[3]) / 2)
    return (x_center, y_center)

cap = DepthCamera()

crop_dir_name = "ultralytics_crop"
if not os.path.exists(crop_dir_name):
    os.mkdir(crop_dir_name)


idx = 0
while True:
    ret, success, im0 = cap.get_frame()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(im0, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    annotator = Annotator(im0, line_width=2, example=names)

    if boxes is not None:
        for box, cls in zip(boxes, clss):
            idx += 1
            annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

            crop_obj = im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
            center_point = get_center_of_bbox(box)

            cv2.circle(im0, center_point, 5, (0, 0, 255), -1)
            distance = success[center_point[1], center_point[0]]
            cv2.putText(im0, "{}mm".format(distance), (center_point[0], center_point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

            cv2.imwrite(os.path.join(crop_dir_name, str(idx) + ".png"), crop_obj)


    cv2.imshow("ultralytics", im0)
    cv2.imshow("depth frame", success)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()