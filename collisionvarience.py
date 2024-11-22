import time
from rplidar import RPLidar

# Constants for LIDAR setup
PORT_NAME = '/dev/ttyUSB0'
BAUD_RATE = 115200
GREEN_ZONE_THRESHOLD = 1.25  # 1.25 meters (green zone is farther than red zone)
MIN_VALID_DISTANCE = 0.1  # Ignore readings below 0.1 meters
MAX_VALID_DISTANCE = 6.0  # Ignore readings above 6 meters
MIN_ANGLE = -15  # Minimum angle for front scan (degrees)
MAX_ANGLE = 15   # Maximum angle for front scan (degrees)

# Deceleration constant (m/s^2)
DECELERATION = 0.2587

def calculate_stopping_distance(initial_velocity):
    # Calculate the stopping distance using d = (u^2) / (2 * a)
    stopping_distance = (initial_velocity ** 2) / (2 * DECELERATION)
    return stopping_distance

def detect_zones():
    lidar = RPLidar(PORT_NAME, baudrate=BAUD_RATE)
    lidar.start_motor()  # Ensure motor is started

    try:
        print("Starting LIDAR scan...")
        time.sleep(2)  # Add a delay to allow the LIDAR to initialize

        # Prompt for initial velocity
        initial_velocity = float(input("Enter the initial velocity in m/s: "))
        stopping_distance = calculate_stopping_distance(initial_velocity)
        print(f"Calculated stopping distance (Red Zone): {stopping_distance:.2f} meters")

        # Variables to track the previous zone status
        last_red_zone = False
        last_green_zone = False

        for scan in lidar.iter_scans():
            red_zone = False
            green_zone = False

            # Iterate through the measurements in the scan
            for (quality, angle, distance) in scan:
                distance_meters = distance / 1000.0  # Convert to meters

                # Filter based on angle and distance range
                if MIN_ANGLE <= angle <= MAX_ANGLE and MIN_VALID_DISTANCE <= distance_meters <= MAX_VALID_DISTANCE:
                    # Zone detection based on calculated stopping distance and green zone
                    if distance_meters <= stopping_distance:
                        red_zone = True
                    elif stopping_distance < distance_meters <= GREEN_ZONE_THRESHOLD:
                        green_zone = True

            # Only print if the zone status has changed
            if red_zone and not last_red_zone:
                print(f"Red zone: Object detected within stopping distance ({stopping_distance:.2f} meters)!")
                last_red_zone = True
                last_green_zone = False
            elif green_zone and not last_green_zone:
                print(f"Green zone: Object detected farther than {stopping_distance:.2f} but within {GREEN_ZONE_THRESHOLD} meters.")
                last_green_zone = True
                last_red_zone = False
            elif not red_zone and not green_zone and (last_red_zone or last_green_zone):
                print("No object detected in defined zones.")
                last_red_zone = False
                last_green_zone = False
                
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        lidar.stop_motor()
        lidar.stop()
        lidar.disconnect()
        print("LIDAR disconnected.")

if __name__ == '__main__':
    detect_zones()

