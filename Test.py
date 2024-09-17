import odrive
from odrive.utils import *
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from tktooltip import ToolTip
import time
import Jetson.GPIO as GPIO

# Function to find and connect to ODrive
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

# Display drive voltage
print("ODrive Voltage:", odrv0.vbus_voltage)

# Initialize GPIOs
pwm = 12
steering = 13
limit2 = 26  # Left limit switch
limit1 = 23  # Right limit switch
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup([pwm, steering], GPIO.OUT, initial=GPIO.LOW)
GPIO.setup([limit1, limit2], GPIO.IN)


# Activate the ODrive
odrv0.axis0.requested_state = 8

for x in (0,1,0.1):

        odrv0.axis0.controller.input_pos = x 


