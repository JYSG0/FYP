import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
from ultralytics.utils.plotting import Annotator, colors
from realsense_depth import DepthCamera
import matplotlib.pyplot as plt
import numpy as np
from source import FastestRplidar
import time

# Initialize global variable
cap = DepthCamera()  # RealSense Camera
model = YOLO("best.pt")  # Load the YOLO model
names = model.names

# Function to get the center of a bounding box
def get_center_of_bbox(box):
    x_center = int((box[0] + box[2]) / 2)
    y_center = int((box[1] + box[3]) / 2)
    return (x_center, y_center)

class Lidar:

    def get_data(self):
        """Fetch scan data from the LiDAR."""
        return self.lidar.get_scan_as_xy(filter_quality=True)

    def stop(self):
        """Stop the LiDAR motor."""
        self.lidar.stopmotor()

    def __init__(self, max_retries=5, retry_delay=2):
        """
        Initialize the Lidar class with retry functionality.
        
        :param max_retries: Maximum number of retry attempts.
        :param retry_delay: Delay (in seconds) between retries.
        """
        self.lidar = FastestRplidar()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.connect_with_retry()

    

    def connect_with_retry(self):
        """Attempt to connect to the LiDAR with retries."""
        attempts = 0
        while attempts < self.max_retries:
            try:
                print(f"Connecting to LiDAR (Attempt {attempts + 1}/{self.max_retries})...")
                self.lidar.connectlidar()
                # Start the LiDAR motor
                self.lidar.startmotor(my_scanmode=2)
                # Verify the connection (example: fetch a sample scan)
                test_data = self.lidar.get_scan_as_xy(filter_quality=False)
                if test_data:  # If data is received, the connection is valid
                    print("Connection successful!")
                    return
                else:
                    raise RuntimeError("Connection verification failed. No data received.")
            
                
            except Exception as e:
                attempts += 1
                print(f"Connection failed: {e}. Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)

        # If all attempts fail, raise an exception
        raise RuntimeError("Failed to connect to LiDAR after multiple attempts.")


def is_point_in_triangle(x, y):
    """Check if a point (x, y) is inside the triangle."""
    # Conditions for the triangle:
    # 1. Below y = 1.2x (Red line)
    # 2. Above y = -1.2x (Purple line)
    # 3. Above y = -1250 (Horizontal black line)
    return y <= 1.2 * x and y <= -1.2 * x and y >= -1250 and y < 0

def detect_zones():
    lidar = Lidar()
            
    # Set up the matplotlib figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(6000, -6000)
    ax.set_ylim(6000, -6000)
    ax.set_xlabel("X-axis (mm)")
    ax.set_ylabel("Y-axis (mm)")
    ax.set_title("Real-Time LiDAR Scan")
    scatter = ax.scatter([], [], s=1)  # Initialize scatter plot

    # Plot line using an expression (e.g., y = 1.2x)
    x_line = np.linspace(-6000, 6000, 500)  # Generate X values from -6000 to 6000
    y_line = 1.2 * x_line  # Calculate Y values for the red line
    ax.plot(x_line, y_line, color='red', linewidth=1, label='y = 1.2x')  

    # Plot line using an expression (e.g., y = -1.2x)
    y_line1 = -1.2 * x_line  # Calculate Y values for the purple line
    ax.plot(x_line, y_line1, color='purple', linewidth=1, label='y = -1.2x ')  

    # Draw a horizontal line at y = -1250
    ax.axhline(y=-1250, color='black', linestyle='--', linewidth=1, label='y = -1250')  

    ax.legend()  # Add a legend to distinguish points and the line

    try:
        while True:
            # Get LiDAR data
            scan_data = lidar.get_data()
            as_np = np.asarray(scan_data)

            # Update plot data
            x_data = -as_np[:, 1]
            y_data = -as_np[:, 0]
            scatter.set_offsets(np.c_[x_data, y_data])

            # Check each point to see if it is inside the triangle
            for x, y in zip(x_data, y_data):
                if is_point_in_triangle(x, y):
                    print(f"Point inside triangle detected: ({x:.2f}, {y:.2f})")

            # Update the plot
            plt.pause(0.1)  # Add a small delay for smooth updating

    except KeyboardInterrupt:
        print("Exiting program...")
    finally:
        # Stop the LiDAR before exiting
        lidar.stop()
        plt.close(fig)

# Function for camera detection
def detection():
    print("Starting YOLO detection...")

    while True:
        ret, success, frame = cap.get_frame()
        if not ret:
            print("Failed to grab frame")
            continue

        # Run YOLO detection
        results = model(frame, conf=0.5, verbose=False, device=0)
        annotated_frame = results[0].plot()
        boxes = results[0].boxes.xyxy.cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        # Annotate the frame with bounding boxes and labels
        annotator = Annotator(annotated_frame, line_width=2, example=names)
        if boxes:
            for box, cls in zip(boxes, clss):
                annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])
                center_point = get_center_of_bbox(box)
                cv2.circle(annotated_frame, center_point, 5, (0, 0, 255), -1)
                distance = success[center_point[1], center_point[0]]
                cv2.putText(annotated_frame, f"{distance}mm", (center_point[0], center_point[1] - 20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        cv2.imshow("YOLOv8 Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Stopping YOLO detection...")
    cap.release()
    cv2.destroyAllWindows()

# Main function to run both threads
if __name__ == '__main__':
    print("Starting concurrent LIDAR and Camera detection...")

    # Start YOLO detection thread
    detection_thread = threading.Thread(target=detection, daemon=True)
    # Start LIDAR detection thread
    lidar_thread = threading.Thread(target=detect_zones, daemon=True)

    # Start both threads
    detection_thread.start()
    lidar_thread.start()

    # Wait for threads to finish (Ctrl+C to stop)
    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Stopping all threads...")
