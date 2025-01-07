import matplotlib.pyplot as plt
import numpy as np
from source import FastestRplidar
import time

class Lidar:
    def __init__(self):
        # Initialize the library and create an object
        self.lidar = FastestRplidar()

        # Connect the LiDAR using the default port (tty/USB0)
        print('Connecting to LiDAR...')
        self.lidar.connectlidar()
        print('Connection successful!')

        # Start the LiDAR motor
        self.lidar.startmotor(my_scanmode=2)

    def get_data(self):
        """Fetch scan data from the LiDAR."""
        return self.lidar.get_scan_as_xy(filter_quality=True)

    def stop(self):
        """Stop the LiDAR motor."""
        self.lidar.stopmotor()

def is_point_in_triangle(x, y):
    """Check if a point (x, y) is inside the triangle."""
    # Conditions for the triangle:
    # 1. Below y = 1.2x (Red line)
    # 2. Above y = -1.2x (Purple line)
    # 3. Above y = -1250 (Horizontal black line)
    return y <= 1.1 * x and y <= -1.1 * x and y >= -1400 and y < 0

def main():
    # Create a Lidar object
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
    y_line = 1.1 * x_line  # Calculate Y values for the red line
    ax.plot(x_line, y_line, color='red', linewidth=1, label='y = 1.1x')  

    # Plot line using an expression (e.g., y = -1.2x)
    y_line1 = -1.1 * x_line  # Calculate Y values for the purple line
    ax.plot(x_line, y_line1, color='purple', linewidth=1, label='y = -1.1x ')  

    # Draw a horizontal line at y = -1250
    ax.axhline(y=-1400, color='black', linestyle='--', linewidth=1, label='y = -1400')  

    ax.legend()  # Add a legend to distinguish points and the line

    try:
        while True:
            # Get LiDAR data
            scan_data = lidar.get_data()
            print(f"Scan data: {scan_data}")            
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
        # Stop the LiDAR before exiting
        lidar.stop()
        plt.close(fig)

if __name__ == "__main__":
    main()
