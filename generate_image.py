import os
import cv2

cap = cv2.VideoCapture(6)

if not cap.isOpened():
    print("Unable to read camera feed")

output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# Get the highest existing image counter in the output directory
existing_images = [f for f in os.listdir(output_dir) if f.startswith("opencv_frame_") and f.endswith(".png")]
if existing_images:
    img_counter = max([int(f.split('_')[2].split('.')[0]) for f in existing_images]) + 1
else:
    img_counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow('Webcam', frame)

    k = cv2.waitKey(1) & 0xFF  # wait for 1ms for key to be pressed (ignore last 8 bits)

    if k % 256 == 27:  # press escape or q to quit
        print("Escape hit, closing...")
        break
    elif k == ord('q'):  # ord('q') gets the ASCII for 'q'
        print("'q' key pressed. Exiting...")
        break
    elif k % 256 == ord('s'):  # 's' key is pressed to save frame
        img_name = os.path.join(output_dir, f"opencv_frame_{img_counter}.png")  # set image name and counter
        cv2.imwrite(img_name, frame)  # writes image file
        print(f"{img_name} written!")
        img_counter += 1

cap.release()
cv2.destroyAllWindows()
