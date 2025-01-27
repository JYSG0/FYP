#Tap a key, move forward 3 secs, 



#Tap d key, move back 3 secs, 








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
is_moving_forward = False
is_reversing = False
running_pwm = False  # Represents whether the system is actively moving
movement_duration = 3  # seconds

# Function to handle key press
def handle_key_press():
    global key_pressed_time, is_moving_forward, running_pwm
    if not is_moving_forward and not is_reversing:  # Only allow triggering when stationary
        key_pressed_time = time.time()
        is_moving_forward = True
        running_pwm = True
        print("Key 'a' pressed: Starting forward movement")
        
# Function to handle key press2
def handle_key_press2():
    global key_pressed_time, is_reversing, running_pwm
    if not is_moving_forward and not is_reversing:  # Only allow triggering when stationary
        key_pressed_time = time.time()
        is_reversing = True
        running_pwm = True
        print("Key 'd' pressed: Starting reverse movement")

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

    # Toggle the pin
    pin.value = True
    time.sleep(high_time)
    pin.value = False
    time.sleep(low_time)

#print("Press 'a' or 'd' to steer left/right. Press ESC to exit.")

# Flags to track key presses
running_pwm = False

# Main control logic
def control_loop():
    global is_moving_forward, is_reversing, running_pwm, key_pressed_time

    while True:
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        # Enable motor
        enable.value = True

        if is_moving_forward:

            if key_pressed_time is not None:  # Check if key_pressed_time is set
                steering.value = False
                running_pwm = True
                elapsed_time = time.time() - key_pressed_time
                if elapsed_time < movement_duration:
                    print(f"Moving forward... Elapsed time: {elapsed_time:.2f} seconds")
                else:
                    # Transition to reverse
                    is_moving_forward = False
                    # Stop after reversing
                    running_pwm = False
                    key_pressed_time = None  # Reset for the next cycle
                    print("Foward motion complete. Stopping.")

        elif is_reversing:
            if key_pressed_time is not None:  # Check if key_pressed_time is set
                elapsed_time = time.time() - key_pressed_time
                if elapsed_time < movement_duration:
                    print(f"Reversing... Elapsed time: {elapsed_time:.2f} seconds")
                    steering.value = True
                    running_pwm = True
                else:
                    # Stop after reversing
                    running_pwm = False
                    is_reversing = False
                    key_pressed_time = None  # Reset for the next cycle
                    print("Reverse motion complete. Stopping.")
        # 'd' is pressed
        if keys[pygame.K_d]:  # moving back
            handle_key_press2()

        # Check for key press
        if keys[pygame.K_a]:
            handle_key_press()
   # Generate PWM signal continuously if a direction is active
        if running_pwm:
            simulate_pwm_continuous(pwm, duty_cycle=50)  # Adjust duty cycle as needed

        # Break the loop if 'ESC'aaaaad
        if keys[pygame.K_ESCAPE]:  # ESC to exit and reset GPIO
            enable.value = False
            pwm.value = False  # Stop the PWM
            break
        time.sleep(0.001)  # Small delay to avoid busy looping

# Run the control loop
print("Press 'a' to start the motion.")
control_loop()
