from rplidar import RPLidar

# Adjust the serial port and baud rate to your specific setup
PORT_NAME = '/dev/ttyUSB0'  # Replace with the correct port for your RPLIDAR
BAUD_RATE = 115200  # Most RPLIDARs use this baud rate, but adjust if needed
RED_ZONE_THRESHOLD = 0.3  # 1 meter
GREEN_ZONE_THRESHOLD = 1  # 5 meters

def detect_zones():
    lidar = RPLidar(PORT_NAME, baudrate=BAUD_RATE)  # Ensuring correct serial port and baud rate

    try:
        print("Starting LIDAR scan...")
        for scan in lidar.iter_scans():
            red_zone = False
            green_zone = False
            
            for (_, _, distance) in scan:
                # Convert distance to meters
                distance_meters = distance / 1000.0

                if distance_meters <= RED_ZONE_THRESHOLD:
                    red_zone = True
                elif distance_meters > GREEN_ZONE_THRESHOLD:
                    green_zone = True

            if red_zone:
                print("Red zone: Object detected within 1 meter!")
            elif green_zone:
                print("Green zone: Object detected farther than 5 meters.")
            else:
                print("No object detected in defined zones.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        lidar.stop()
        lidar.disconnect()
        print("LIDAR disconnected.")

if __name__ == '__main__':
    detect_zones()

