import sys
import odrive
import asyncio
import json
from fastapi import FastAPI, WebSocket
from uvicorn import Config, Server
import threading
import time
import keyboard

import Jetson.GPIO as GPIO
import busio
import board
import digitalio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# Globals
auto = False  # Manual mode by default
input_velocity = 1  # Default velocity
routeActive = False
connected_clients = []  # Store WebSocket clients
dir, angleToTurn, azimuth, turn = None, None, None, None  # Data from ESP32
speedMode = False
input_velocity = 1
within_tolerance = False
brozimuth = None

start_mode = True
steering_angle = None

i2c=busio.I2C(board.SCL_1, board.SDA_1)
time.sleep(0.1)  # Small delay to stabilize the connection
ads=ADS.ADS1115(i2c, address=0x48)

chan1 = AnalogIn(ads, ADS.P0)
chan2=AnalogIn(ads, ADS.P3)
print(chan1.value, chan1.voltage)       
print(chan2.value, chan2.voltage)

#Initialise GPIO Pins
pwm = digitalio.DigitalInOut(board.D22)
steering = digitalio.DigitalInOut(board.D13)
pwm.direction = digitalio.Direction.OUTPUT
steering.direction = digitalio.Direction.OUTPUT

#Initial Low 
pwm.value = False  # Set GPIO12 HIGH
steering.value = False  # Set GPIO13 HIGH

# FastAPI app
app = FastAPI()

# Function to connect to ODrive
def connect_to_odrive():
    try:
        odrv = odrive.find_any()
        if odrv is None:
            print("ODrive not found. Exiting.")
            sys.exit()
        return odrv
    except Exception as e:
        print(f"Error connecting to ODrive: {e}")
        sys.exit()

# Initialize ODrive
odrv0 = connect_to_odrive()
odrv0.axis1.requested_state = 8
odrv0.axis0.requested_state = 8

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global dir, angleToTurn, turn, routeActive, minBrozimuth, maxBrozimuth, within_tolerance, brozimuth
    await websocket.accept()
    connected_clients.append(websocket)
    print("WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                print(message)
                message = json.loads(message)
                if message.get("type") == "vehicleControl":
                    routeActive = True
                    dir = message.get("jetson")
                    turn = message.get("modifier")
                    angleToTurn = message.get("angleToTurn")
                    within_tolerance = message.get("within_tolerance")

                    dir = "left"
                    angleToTurn = -50

                    brozimuth = map_value(angleToTurn, -180, 180, -40, 40)
                    minBrozimuth = brozimuth - 3.5
                    maxBrozimuth = brozimuth + 3.5

                    print("brozimuth: ", brozimuth) # Angle to Turn but scaled to vehicle steering angle
                    print("Direction to turn to: ", dir) #direction to move
                    print("turn: ", turn)
                    print(f"Turn {angleToTurn} degrees") # Required Angle to turn for whole vehicle

            except json.JSONDecodeError:
                print("Invalid JSON received")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        connected_clients.remove(websocket)
        print("WebSocket client disconnected")

def map_value(value, input_min, input_max, output_min, output_max):
    return ((value - input_min) * (output_max - output_min)) / (input_max - input_min) + output_min

# Manual control
def manual_control():
    global input_velocity, start_mode
    print("Manual mode: vehicle stopped")
    odrv0.axis1.controller.input_vel = 0
    odrv0.axis0.controller.input_vel = 0
    # Handle specific key presses
    if keyboard.is_pressed('e'):
        start_mode = not start_mode

        if start_mode:  # Start motors
            print("Starting...")
            odrv0.axis0.requested_state = 8  # Start Motor
            odrv0.axis1.requested_state = 8  # Start Motor
        else:  # Stop motors
            print("Stopping...")
            odrv0.axis1.requested_state = 1  # Set ODrive to idle state
            odrv0.axis0.requested_state = 1  # Set ODrive to idle state

    # if 0 in class_ids:
    #     print("Go")
    #     odrv0.axis1.controller.input_vel = 1
    #     odrv0.axis0.controller.input_vel = -1

    if keyboard.is_pressed('w'):
        odrv0.axis1.controller.input_vel = input_velocity
        odrv0.axis0.controller.input_vel = -input_velocity

        if keyboard.is_pressed('a'):
            odrv0.axis1.controller.input_vel = input_velocity / 2
            odrv0.axis0.controller.input_vel = -input_velocity

        elif keyboard.is_pressed('d'):
            odrv0.axis1.controller.input_vel = input_velocity
            odrv0.axis0.controller.input_vel = -input_velocity / 2

        steer()
   
        # if odrv0.axis1.controller.input_vel >= 2:

        #     if people_detect:
        #         # People or Stop sign detected
        #         if 4 in class_ids or 8 in class_ids:
        #                 print("Stop")
        #                 odrv0.axis1.controller.input_vel = 0
        #                 odrv0.axis0.controller.input_vel = 0

        #     # Slow or Speed sign or Hump or Pedestrain Sign detected
        #     elif 6 in class_ids or 7 in class_ids or 1 in class_ids or 3 in class_ids:
        #         print("Slow")
        #         odrv0.axis1.controller.input_vel = 0.8
        #         odrv0.axis0.controller.input_vel = -0.8

    if keyboard.is_pressed('s'):
        odrv0.axis1.controller.input_vel = -input_velocity
        odrv0.axis0.controller.input_vel = input_velocity
        print(f"BACKWARD at speed: {input_velocity}")

    # if keyboard.is_pressed('o'):
    #     people_detect = not people_detect
    #     time.sleep(0.5)
    #     if people_detect:  # Start people_detection
    #         print("Detection On")

    #     else:  # Stop people_detection
    #          print("Detection Off")

    # 'shift' key to brake
    if keyboard.is_pressed('shift'):
        odrv0.axis1.controller.input_vel = 0
        odrv0.axis0.controller.input_vel = 0
        input_velocity = 1
        
        print ('STOP')

# Auto control
def auto_control():
    global dir, turn, input_velocity
    print("Auto mode: Processing ESP32 data")

    #Axis 1 = right wheel
    #Axis 0 = left wheel

    time.sleep(1)

    if routeActive:
        if dir == "straight":
            odrv0.axis1.controller.input_vel = input_velocity
            odrv0.axis0.controller.input_vel = -input_velocity

        elif dir == "left":
            print("Left")
            odrv0.axis1.controller.input_vel = input_velocity
            odrv0.axis0.controller.input_vel = -input_velocity/2

        elif dir == "right":
            print("right")
            odrv0.axis1.controller.input_vel = input_velocity/2
            odrv0.axis0.controller.input_vel = -input_velocity

        steer()
    else:
        odrv0.axis1.controller.input_vel = 0
        odrv0.axis0.controller.input_vel = 0

def steer():
    global minBrozimuth, maxBrozimuth, brozimuth
    pot_value = map_value (chan1.value, 0, 26230, 0, 1023)
    steering_angle = map_value(pot_value, 0 , 1023, -40 ,40)

    if auto:
        print("auto: ", steering_angle)

        # if minBrozimuth <= steering_angle <= maxBrozimuth:  #If steering angle is within turn range

        #If exceed limit, set to limit
        if brozimuth > 15:
            brozimuth = 15
        elif brozimuth <-6:
            brozimuth = -6

        print("In brozimuth range")
        pwm.value = True  # Set GPIO12 high
        steering.value = True  # Set GPIO13 low

        if minBrozimuth <= steering_angle <= maxBrozimuth:
            print("In steering range")
            pwm.value = False
            steering.value = False
        else:
            if steering_angle <= brozimuth:  # Steer Left
                pwm.value = True
                steering.value = False
                if pot_value is not None:
                    print(f"Steering Left: Potentiometer Value: {int(steering_angle)}")
            elif steering_angle > brozimuth:    #Steer right
                pwm.value = True
                steering.value = True
                if pot_value is not None:
                    print(f"Steering Right: Potentiometer Value: {int(steering_angle)}")
            else:   #Steer right limit
                pwm.value = False
                steering.value = False
                if pot_value is not None:
                    print(f"No Steering: Potentiometer Value: {int(steering_angle)}")

    elif not auto:
        print("Manual")
        # 'a' is pressed
        if keyboard.is_pressed('a'):
            print("Steer left", steering_angle)
            if steering_angle <= 15:  # Steer Left
                pwm.value = True
                steering.value = False
                if pot_value is not None:
                    print(f"Steering Left: Potentiometer Value: {int(steering_angle)}")
            else:   #Steer right limit
                pwm.value = False
                steering.value = False
                if pot_value is not None:
                    print(f"No Steering: Potentiometer Value: {int(steering_angle)}")
        
        elif keyboard.is_pressed('d'):
            print("Steer right", steering_angle)
            if steering_angle >= -6:    #Steer right
                pwm.value = True
                steering.value = True
                if pot_value is not None:
                    print(f"Steering Right: Potentiometer Value: {int(steering_angle)}")
            else:   #Steer left limit
                pwm.value = False
                steering.value = False
                if pot_value is not None:
                    print(f"No Steering: Potentiometer Value: {int(steering_angle)}")

        else: # Joystick IDLE
            pwm.value = False
            steering.value = False
            if pot_value is not None:
                print(f"No Steering: Potentiometer Value: {int(steering_angle)}")
    
    else:
        pwm.value = False
        steering.value = False
        if pot_value is not None:
            print(f"No Steering: Potentiometer Value: {int(steering_angle)}")
    
    if keyboard.is_pressed('esc'):
        pwm.value = False
        steering.value = False

# Toggle between manual and auto
def toggle_control():
    global auto, input_velocity, steering_angle, speedMode
    while True:

        if keyboard.is_pressed('o'):
            auto = not auto
            print(f"Switched to {'Auto' if auto else 'Manual'} mode")

        if keyboard.is_pressed('space'):
            speedMode = not speedMode

            if speedMode:
                input_velocity = 2
                print(f"Increased velocity to: {input_velocity}")
            else:
                input_velocity = 1
                print(f"Velocity to: {input_velocity}")

        if auto:
            auto_control()
        else:
            manual_control()

        time.sleep(0.1)
# Main function
async def main():
    toggle_thread = threading.Thread(target=toggle_control, daemon=True)
    toggle_thread.start()

    # Start the FastAPI server
    config = Config(app, host="127.0.0.1", port=8765)
    server = Server(config)
    await server.serve()

if __name__ == '__main__':
    asyncio.run(main())
