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
is_moving_forward = False
is_reversing = False
motion_completed = False  # Ensures motion runs only once per cycle

# Function to simulate PWM
def simulate_pwm(pin, duration, duty_cycle, frequency=1000):
    """
    Simulates PWM by toggling the pin on and off for a specific duration.
    :param pin: The digitalio pin to toggle.
    :param duration: Duration to run the PWM in seconds.
    :param duty_cycle: Duty cycle as a percentage (0-100).
    :param frequency: PWM frequency in Hz (default: 1000).
    """
    period = 1 / frequency
    high_time = period * (duty_cycle / 100)
    low_time = period - high_time

    start_time = time.time()
    while time.time() - start_time < duration:
        pin.value = True
        time.sleep(high_time)
        pin.value = False
        time.sleep(low_time)

# Main control logic
def control_loop():
    global is_moving_forward, is_reversing, motion_completed

    while True:
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        # Enable motor
        enable.value = True

        # If 'a' is pressed and no motion is currently in progress
        if keys[pygame.K_a] and not is_moving_forward and not is_reversing and not motion_completed:
            print("Key 'a' pressed: Moving forward for 3 seconds")
            is_moving_forward = True
            steering.value = False  # Forward direction
            simulate_pwm(pwm, duration=3, duty_cycle=50)  # Run forward for 3 seconds
            print("Forward motion complete.")
            is_moving_forward = False
            is_reversing = True  # Prepare for reverse motion after release

        # If 'a' is released and reversing is pending
        if not keys[pygame.K_a] and is_reversing:
            print("Key 'a' released: Moving backward for 3 seconds")
            steering.value = True  # Reverse direction
            simulate_pwm(pwm, duration=3, duty_cycle=50)  # Run backward for 3 seconds
            print("Reverse motion complete.")
            is_reversing = False
            motion_completed = True  # Mark the motion cycle as complete

        # Reset motion cycle if no key is pressed and motion is completed
        if not keys[pygame.K_a] and motion_completed:
            print("Motion cycle complete. Waiting for next input.")
            motion_completed = False  # Allow new motion cycle to start

        # Exit if 'ESC' is pressed
        if keys[pygame.K_ESCAPE]:
            enable.value = False
            pwm.value = False  # Stop the PWM
            break

        time.sleep(0.001)  # Small delay to avoid busy looping

# Run the control loop
try:
    print("Hold 'a' to move forward for 3 seconds. Release 'a' to move backward for 3 seconds. Press ESC to exit.")
    control_loop()
finally:
    pwm.value = False
    enable.value = False
    steering.value = False
    pygame.quit()
    print("Exited successfully.")

