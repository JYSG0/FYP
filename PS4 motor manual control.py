import time
import pygame
import odrive
import Jetson.GPIO as GPIO  # Ensure Jetson.GPIO is imported

# Initialize Pygame for controller input
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

print(f"Controller detected: {joystick.get_name()}")

# ODrive connection setup
def connect_to_odrive():
    try:
        odrv = odrive.find_any()
        if odrv is None:
            print("ODrive not found. Exiting program.")
            exit()
        print("ODrive connected successfully!")
        return odrv
    except Exception as e:
        print("Error connecting to ODrive:", e)
        exit()

# Connect to ODrive
odrv0 = connect_to_odrive()

# GPIO setup for controlling external components (if needed)
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Example GPIO pins for PWM or other external controls (update with your pin numbers)
pwm_pin = 12
steering_pin = 13

# Set GPIO pins as output
GPIO.setup([pwm_pin, steering_pin], GPIO.OUT, initial=GPIO.LOW)

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
    GPIO.output(pwm_pin, GPIO.HIGH)  # Example of controlling an external GPIO pin (PWM)
    print(f"Motor moving forward to position: {current_position}")

# Function to stop the motor or move it backward
def move_motor_backward():
    global current_position
    current_position -= position_increment  # Decrease position by a small value
    odrv0.axis0.controller.input_pos = current_position  # Update motor position
    GPIO.output(pwm_pin, GPIO.LOW)  # Example of controlling an external GPIO pin (PWM)
    print(f"Motor moving backward to position: {current_position}")

# Loop to get controller events and control the motor
while True:
    for event in pygame.event.get():
        # Debugging: Print the raw axis values for L2 and R2
        right_trigger_value = joystick.get_axis(right_trigger_axis)
        left_trigger_value = joystick.get_axis(left_trigger_axis)
        print(f"R2 (Right Trigger) Value: {right_trigger_value}, L2 (Left Trigger) Value: {left_trigger_value}")

        # Handle trigger events
        if right_trigger_value > trigger_threshold:
            move_motor_forward()
        elif left_trigger_value > trigger_threshold:
            move_motor_backward()

    # Limit the loop to a reasonable refresh rate
    time.sleep(0.1)
