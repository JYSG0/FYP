from rplidar import RPLidar

# Adjust the serial port and baud rate to your specific setup
PORT_NAME = '/dev/ttyUSB0'  # Replace with the correct port for your RPLIDAR
BAUD_RATE = 115200  # Adjust if needed

def stop_lidar():
    lidar = RPLidar(PORT_NAME, baudrate=BAUD_RATE)  # Ensure correct serial port and baud rate
    try:
        # Stopping the motor immediately
        print("Stopping the LIDAR motor...")
        lidar.stop_motor()  # Explicitly stop the motor from spinning
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        lidar.disconnect()  # Ensure the LIDAR is disconnected
        print("LIDAR disconnected.")

if __name__ == '__main__':
    stop_lidar()

