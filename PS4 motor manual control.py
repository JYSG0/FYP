import time
import pygame
import odrive

# Initialize Pygame
pygame.init()

# Initialize the joystick module
pygame.joystick.init()

# Check if the PS4 controller is connected
joystick_count = pygame.joystick.get_count()
if joystick_count == 0:
    print("No controller detected. Please connect your PS4 controller.")
    exit()

# Select the first controller
joystick = pygame.joystick.Joystick(0)
joystick.init()

# ODrive connection setup
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

# Connect to ODrive
odrv0 = connect_to_odrive()

# PS4 controller trigger axes:
left_trigger_axis = 4  # L2 Trigger (Brake/Decrease Position)
right_trigger_axis = 5  # R2 Trigger (Throttle/Increase Position)

# Motor control parameters
position_increment = 0.05  # Smaller increment for position control
current_position = 0.0

# Joystick axis threshold for movement sensitivity
trigger_threshold = 0.1

# Function to move motor forward (using position control)
def move_motor_forward():
    global current_position
    current_position += position_increment  # Increment position by a small value
    odrv0.axis0.controller.input_pos = current_position  # Update motor position
    print(f"Motor moving forward to position: {current_position}")

# Function to stop the motor or move it backward
def move_motor_backward():
    global current_position
    current_position -= position_increment  # Decrease position by a small value
    odrv0.axis0.controller.input_pos = current_position  # Update motor position
    print(f"Motor moving backward to position: {current_position}")

# Loop to get controller events and control the motor
while True:
    for event in pygame.event.get():
        # Handle trigger events
        if event.type == pygame.JOYAXISMOTION:
            # Check the right trigger (R2) axis for motor control (Throttle)
            if event.axis == right_trigger_axis:
                trigger_value = (event.value + 1) / 2  # Convert value from [-1, 1] to [0, 1]
                if trigger_value > trigger_threshold:  # Significant trigger press for throttle
                    move_motor_forward()

            # Check the left trigger (L2) axis for stopping the motor (Brake)
            elif event.axis == left_trigger_axis:
                trigger_value = (event.value + 1) / 2  # Convert value from [-1, 1] to [0, 1]
                if trigger_value > trigger_threshold:  # If L2 is pressed beyond the threshold
                    move_motor_backward()

    # Limit the loop to a reasonable refresh rate
    time.sleep(0.1)
