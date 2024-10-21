import Jetson.GPIO as GPIO
import pygame
import busio
import board
import time
import digitalio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# Print out the available pins in the board module
# Print out the pins for SDA and SCL
print(f"SDA pin: {board.SDA_1}")
print(f"SCL pin: {board.SCL_1}")
i2c=busio.I2C(board.SCL_1, board.SDA_1)
ads=ADS.ADS1115(i2c, address=0x48)

chan1 = AnalogIn(ads, ADS.P0)
chan2=AnalogIn(ads, ADS.P3)
print(chan1.value, chan1.voltage)
print(chan2.value, chan2.voltage)

# Function to map chan1 value to a specific range (e.g., 10 to 100)
def map_potentiometer_value(value, input_min=0, input_max=26230, output_min=0, output_max=1023):
    return ((value - input_min) * (output_max - output_min)) / (input_max - input_min) + output_min

# Initialize pygame
pygame.init()

#Initialise GPIO Pins
pwm = digitalio.DigitalInOut(board.D12)
steering = digitalio.DigitalInOut(board.D13)
pwm.direction = digitalio.Direction.OUTPUT
steering.direction = digitalio.Direction.OUTPUT

#Initial Low 
pwm.value = False  # Set GPIO12 high
steering.value = False  # Set GPIO13 low

# Define window size (To capture events)
screen = pygame.display.set_mode((640,480))

while True:
        pygame.event.pump()
        keys = pygame.key.get_pressed()

          # Read potentiometer value
        try:
            pot_value = map_potentiometer_value(chan1.value)
        except OSError as e:
            print(f"Error reading potentiometer: {e}")
            pot_value = None  # Set to None or handle gracefully if there's an issue
    
                #'a' is pressed
        if keys[pygame.K_a]:  # Steer Left
            pwm.value = False
            steering.value = True
            if pot_value is not None:
                print(f"Steering Left: Potentiometer Value: {pot_value}")
                print(chan1.value, chan1.voltage)

        #'d' is pressed
        if keys[pygame.K_d]:  # Steer Right
            pwm.value = False
            steering.value = False
            if pot_value is not None:
                print(f"Steering Right: Potentiometer Value: {pot_value}")

        if not keys[pygame.K_a] and not keys[pygame.K_d] or pot_value <= 300:  # Joystick IDLE
            pwm.value = True
            steering.value = True
            if pot_value is not None:
                print(f"No Steering: Potentiometer Value: {pot_value}")

        # Break the loop if 'ESC'
        if keys[pygame.K_ESCAPE]:  # ESC to exit and restart motor
            break
