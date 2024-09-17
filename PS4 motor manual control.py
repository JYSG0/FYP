import pygame
import time
import odrive
import Jetson.GPIO as GPIO

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
left_trigger_axis = 4  # L2 Trigger
right_trigger_axis = 5  # R2 Trigger

# Motor control parameters
throttle_percentage = 0
motor_moving = False

# Joystick axis threshold for movement sensitivity
trigger_threshold = 0.1

# GPIO setup for PWM control
pwm = 12  # Motor control pin
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(pwm, GPIO.OUT, initial=GPIO.LOW)

# Function to move motor forward
def move_motor_forward(throttle):
    global motor_moving
    odrv0.axis0.controller.input_pos += throttle  # Increase motor position by throttle value
    motor_moving = True
    print(f"Motor moving forward with throttle: {throttle}")

# Function to stop the motor
def stop_motor():
    global motor_moving
    odrv0.axis0.controller.input_pos = 0  # Stop motor by setting position to 0
    motor_moving = False
    print("Motor stopped")

# Loop to get controller events and control the motor
while True:
    for event in pygame.event.get():
        # Handle trigger events
        if event.type == pygame.JOYAXISMOTION:
            # Check the right trigger (R2) axis for motor control
            if event.axis == right_trigger_axis:
                trigger_value = event.value  # Value between 0.0 (not pressed) and 1.0 (fully pressed)
                if trigger_value > trigger_threshold:  # Significant trigger press for throttle
                    throttle_percentage = int(trigger_value * 100)  # Scale to 0-100%
                    move_motor_forward(throttle_percentage)
                else:
                    throttle_percentage = 0  # No throttle when trigger is not pressed
                    if motor_moving:
                        stop_motor()

            # Check the left trigger (L2) axis for stopping the motor
            elif event.axis == left_trigger_axis:
                if event.value > trigger_threshold:  # If L2 is pressed, stop the motor
                    stop_motor()

    # Limit the loop to a reasonable refresh rate
    time.sleep(0.1)
