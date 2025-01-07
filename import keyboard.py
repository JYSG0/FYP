import keyboard
import digitalio
import board
import time

# Pin Definitions
PWM_PIN = board.D22  # Replace with your actual PWM-capable pin if needed
STEERING_PIN = board.D13  # Replace with your actual steering pin
ENABLE_PIN = board.D6  # Replace with your actual enable pin

# Set up GPIO pins using digitalio
pwm = digitalio.DigitalInOut(PWM_PIN)
pwm.direction = digitalio.Direction.OUTPUT

steering = digitalio.DigitalInOut(STEERING_PIN)
steering.direction = digitalio.Direction.OUTPUT

enable = digitalio.DigitalInOut(ENABLE_PIN)
enable.direction = digitalio.Direction.OUTPUT

# Function to simulate PWM
def simulate_pwm_continuous(pin, duty_cycle, frequency=500):
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

print("Press 'a' or 'd' to steer left/right. Press ESC to exit.")

# Flags to track key presses
running_pwm = False

try:
    while True:

        # Enable motor
        enable.value = True

        # 'a' is pressed
        if keyboard.is_pressed('a'):
            steering.value = False
            running_pwm = True
            print("Steering Left")

        # 'd' is pressed
        elif keyboard.is_pressed('d'):
                steering.value = True
                running_pwm = True
                print("Steering Right")

        # If no keys are pressed, stop steering
        else:
            if running_pwm:  # Stop PWM only once
                pwm.value = False
                running_pwm = False
                print("No steering input - PWM stopped")

        # Generate PWM signal continuously if a direction is active
        if running_pwm:
            simulate_pwm_continuous(pwm, duty_cycle=50)  # Adjust duty cycle as needed

        # Break the loop if 'ESC'aaaaad
        elif keyboard.is_pressed('esc'):
            enable.value = False
            pwm.value = False  # Stop the PWM
            break

        time.sleep(0.001)  # Small delay to reduce CPU usage

except KeyboardInterrupt:
    print("Exiting program...")

finally:
    # Cleanup GPIO resources
    pwm.value = False
    enable.value = False
    steering.value = False
    print("Exited successfully.")
