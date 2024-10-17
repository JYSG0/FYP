import time
from rplidar import RPLidar

# Adjust the serial port and baud rate to your specific setup
PORT_NAME = '/dev/ttyUSB0'
BAUD_RATE = 115200
RED_ZONE_THRESHOLD = 1.2  # 0.9 meters (red zone)
GREEN_ZONE_THRESHOLD = 1.25  # 1 meter (green zone)
MIN_VALID_DISTANCE = 0.1  # Ignore readings below 0.1 meters
MAX_VALID_DISTANCE = 6.0  # Ignore readings above 6 meters (adjust as per your environment)

def detect_zones():
    lidar = RPLidar(PORT_NAME, baudrate=BAUD_RATE)
    lidar.start_motor()  # Ensure motor is started

    try:
        print("Starting LIDAR scan...")
        time.sleep(2)  # Add a 2-second delay to allow the LIDAR to initialize

        for scan in lidar.iter_scans():
            red_zone = False
            green_zone = False

            # Iterate through the measurements in the scan
            for (_, _, distance) in scan:
                distance_meters = distance / 1000.0  # Convert to meters

                # Apply distance filtering to skip noisy/invalid readings
                if MIN_VALID_DISTANCE <= distance_meters <= MAX_VALID_DISTANCE:
                    # Zone detection based on raw distance
                    if distance_meters <= RED_ZONE_THRESHOLD:
                        red_zone = True
                    elif distance_meters > GREEN_ZONE_THRESHOLD:
                        green_zone = True

            # Display zone status without spamming terminal
            if red_zone:
                print(f"Red zone: Object detected within {RED_ZONE_THRESHOLD} meters!")
            elif green_zone:
                print(f"Green zone: Object detected farther than {GREEN_ZONE_THRESHOLD} meters.")
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
