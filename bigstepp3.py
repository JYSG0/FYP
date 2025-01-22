import time
import pygame
import digitalio
import board

# Initialize pygame
pygame.init()

# Pin Definitions
PWM_PIN = board.D27  # Replace with your actual PWM-capable pin if needed
STEERING_PIN = board.D12  # Replace with your actual steering pin
ENABLE_PIN = board.D6  # Replace with your actual enable pin

# Set up GPIO pins using digitalio
pwm = digitalio.DigitalInOut(PWM_PIN)
pwm.direction = digitalio.Direction.OUTPUT

steering = digitalio.DigitalInOut(STEERING_PIN)
steering.direction = digitalio.Direction.OUTPUT

enable = digitalio.DigitalInOut(ENABLE_PIN)
enable.direction = digitalio.Direction.OUTPUT

# Define window size (To capture events)
screen = pygame.display.set_mode((640, 480))

# State variables
key_pressed_time = None
key_held_duration = 0
is_moving_forward = False
is_reversing = False
running_pwm = False  # Represents whether the system is actively moving
hold_started = False

# Function to simulate PWM
def simulate_pwm_continuous(pin, duty_cycle, frequency=1000):
    """
    Simulates PWM by toggling the pin on and off continuously.
    :param pin: The digitalio pin to toggle.
    :param duty_cycle: Duty cycle as a percentage (0-100).
    :param frequency: PWM frequency in Hz (default: 50).
    """
    period = 1 / frequency
    high_time = period * (duty_cycle / 100)
    low_time = period - high_time

    pin.value = True
    time.sleep(high_time)
    pin.value = False
    time.sleep(low_time)

# Main control logic
def control_loop():
    global is_moving_forward, is_reversing, running_pwm, key_pressed_time, key_held_duration, hold_started

    while True:
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        # Enable motor
        enable.value = True

        # If 'a' is pressed
        if keys[pygame.K_a]:
            if not is_moving_forward:  # Detect first press
                print("Key 'a' pressed: Moving forward")
                is_moving_forward = True
                key_pressed_time = time.time()  # Start tracking press duration
                hold_started = True  # Indicates key hold has started

        # If 'a' is released
        else:
            if is_moving_forward and hold_started:  # Ensure it was a valid hold
                key_held_duration = time.time() - key_pressed_time  # Calculate hold duration
                print(f"Key 'a' released: Held for {key_held_duration:.2f} seconds")
                print("Starting reverse motion")
                is_moving_forward = False
                is_reversing = True
                key_pressed_time = time.time()  # Reset timer for reverse
                hold_started = False

        # Forward motion while 'a' is held
        if is_moving_forward:
            steering.value = False
            running_pwm = True
            print("Holding 'a': Moving forward")

        # Reverse motion after 'a' is released
        if is_reversing:
            elapsed_time = time.time() - key_pressed_time if key_pressed_time else 0
            if elapsed_time < key_held_duration:  # Reverse for the same duration as key hold
                print(f"Reversing... Elapsed time: {elapsed_time:.2f} seconds")
                steering.value = True
                running_pwm = True
            else:
                print("Reverse motion complete. Stopping.")
                is_reversing = False
                running_pwm = False

        # Generate PWM signal continuously if running
        if running_pwm:
            simulate_pwm_continuous(pwm, duty_cycle=50)  # Adjust duty cycle as needed
        else:
            pwm.value = False  # Stop PWM if no motion


        if keys[pygame.K_d]:  # Steer Right
            steering.value = True
            running_pwm = True
            print("Steering Right")

        # Exit if 'ESC' is pressed
        if keys[pygame.K_ESCAPE]:
            enable.value = False
            pwm.value = False  # Stop the PWM
            break

        time.sleep(0.001)  # Small delay to avoid busy looping

# Run the control loop
try:
    print("Hold 'a' to move forward. Release 'a' to reverse. Press ESC to exit.")
    control_loop()
finally:
    pwm.value = False
    enable.value = False
    steering.value = False
    pygame.quit()
    print("Exited successfully.")