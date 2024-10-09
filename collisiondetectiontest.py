import time
from rplidar import RPLidar
from collections import deque
import numpy as np  # We need numpy to compute the median

# Adjust the serial port and baud rate to your specific setup
PORT_NAME = '/dev/ttyUSB0'
BAUD_RATE = 115200
RED_ZONE_THRESHOLD = 0.9  # 0.9 meters (red zone)
GREEN_ZONE_THRESHOLD = 1.0  # 1 meter (green zone)
MIN_VALID_DISTANCE = 0.1  # Ignore readings below 0.1 meters
MAX_VALID_DISTANCE = 6.0  # Ignore readings above 6 meters (adjust as per your environment)
BUFFER_SIZE = 5  # Number of recent readings to store for median filter

def detect_zones():
    lidar = RPLidar(PORT_NAME, baudrate=BAUD_RATE)
    lidar.start_motor()  # Ensure motor is started
    recent_distances = deque(maxlen=BUFFER_SIZE)  # Buffer to store recent distance readings

    try:
        print("Starting LIDAR scan...")
        time.sleep(2)  # Add a 2-second delay to allow the LIDAR to initialize

        for scan in lidar.iter_scans():
            red_zone = False
            green_zone = False

            # Iterate through the measurements in the scan
            for (_, _, distance) in scan:
                distance_meters = distance / 1000.0

                # Apply distance filtering to skip noisy/invalid readings
                if MIN_VALID_DISTANCE <= distance_meters <= MAX_VALID_DISTANCE:
                    recent_distances.append(distance_meters)  # Store valid distances in the buffer

            # Apply median filter: Calculate the median of the recent distance readings
            if len(recent_distances) > 0:
                median_distance = np.median(recent_distances)
            else:
                median_distance = float('inf')  # If no valid readings, set to infinity

            # Zone detection based on the median distance
            if median_distance <= RED_ZONE_THRESHOLD:
                red_zone = True
            elif median_distance > GREEN_ZONE_THRESHOLD:
                green_zone = True

            # Display zone status
            if red_zone:
                print("Red zone: Object detected within 0.9 meters!")
            elif green_zone:
                print("Green zone: Object detected farther than 1 meter.")
            else:
                print("No object detected in defined zones.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        lidar.stop_motor()
        lidar.stop()
        lidar.disconnect()
        print("LIDAR disconnected.")

if __name__ == '__main__':
    detect_zones()
