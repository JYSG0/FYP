import Jetson.GPIO as GPIO
import pygame
import busio
import board
import time
import digitalio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# # Print out the available pins in the board module
# # Print out the pins for SDA and SCL
# def list_board_pins():

#     # List all GPIO pins available through the board module
#     pins = [attr for attr in dir(board) if not attr.startswith("_")]
#     print("Available GPIO Pins via the `board` module:")
#     for pin in pins:
#         print(pin)

# # Call the function to print pins
# list_board_pins()

i2c=busio.I2C(board.SCL_1, board.SDA_1)
time.sleep(0.1)  # Small delay to stabilize the connection
ads=ADS.ADS1115(i2c, address=0x48)

chan1 = AnalogIn(ads, ADS.P0)
chan2=AnalogIn(ads, ADS.P3)
print(chan1.value, chan1.voltage)       
print(chan2.value, chan2.voltage)



def map_value(value, input_min, input_max, output_min, output_max):
    return ((value - input_min) * (output_max - output_min)) / (input_max - input_min) + output_min


# Initialize pygame
pygame.init()

#Initialise GPIO Pins
pwm = digitalio.DigitalInOut(board.D22)
steering = digitalio.DigitalInOut(board.D13)
pwm.direction = digitalio.Direction.OUTPUT
steering.direction = digitalio.Direction.OUTPUT

#Initial Low 
pwm.value = False  # Set GPIO12 HIGH
steering.value = False  # Set GPIO13 HIGH

# Define window size (To capture events)
screen = pygame.display.set_mode((640,480))

while True:  
        
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        #   Read potentiometer value
        try:
            pot_value = map_value (chan1.value, 0, 26230, 0, 1023)
            steering_angle = map_value(pot_value, 0 , 1023, -40 ,40)
        except OSError as e:
            print(f"Error reading potentiometer: {e}")
    
        # 'a' is pressed
        if keys[pygame.K_a] and steering_angle <= 39:  # Steer Left
            pwm.value = True
            steering.value = False
            if pot_value is not None:
                print(f"Steering Left: Potentiometer Value: {int(steering_angle)}")
                

        #'d' is pressed
        if keys[pygame.K_d] and steering_angle >= -25:  # Steer Right
            pwm.value = True
            steering.value = True
            if pot_value is not None:
                print(f"Steering Right: Potentiometer Value: {int(steering_angle)}")

        if not keys[pygame.K_a] and not keys[pygame.K_d]: # Joystick IDLE
            pwm.value = False
            steering.value = False
            if pot_value is not None:
                print(f"No Steering: Potentiometer Value: {int(steering_angle)}")
                
        if steering_angle >= 39 and keys[pygame.K_a]:  #Steer Right Limit
            pwm.value = False
            steering.value = False
            if pot_value is not None:
                print(f"No Steering: Potentiometer Value: {int(steering_angle)}")

        if steering_angle <= -25 and keys[pygame.K_d]: #Steer Left Limit
            pwm.value = False
            steering.value = False
            if pot_value is not None:
                print(f"No Steering: Potentiometer Value: {int(steering_angle)}")

    
        # #'a' is pressed
        # if keys[pygame.K_a]:  # Steer Left
        #     pwm.value = True
        #     steering.value = False
        #     if pot_value is not None:
        #         print(f"Steering Left: Potentiometer Value: {int(steering_angle)}")
                

        # #'d' is pressed
        # if keys[pygame.K_d]:  # Steer Right
        #     pwm.value = True
        #     steering.value = True
        #     if pot_value is not None:
        #         print(f"Steering Right: Potentiometer Value: {int(steering_angle)}")

        # if not keys[pygame.K_a] and not keys[pygame.K_d]: # Joystick IDLE
        #     pwm.value = False
        #     steering.value = False
        #     if pot_value is not None:
        #         print(f"No Steering: Potentiometer Value: {int(steering_angle)}")

        if pot_value == None:
            continue
        # Break the loop if 'ESC'
        if keys[pygame.K_ESCAPE]:  # ESC to exit and restart motor
            pwm.value = False
            steering.value = False
            break
