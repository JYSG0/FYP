import odrive
import pygame
import sys
import time
# Initialize pygame
pygame.init()

# Define window size (To capture events)
screen = pygame.display.set_mode((640,480))

# Connect to ODrive
def connect_to_odrive():
    print("Finding odrive")
    try:
        odrv = odrive.find_any()
        if odrv is None:
            print("ODrive not found. Exiting program.")
            exit()
        print("ODrive connected successfully.")
        return odrv
    except Exception as e:
        print("Error connecting to ODrive:", e)
        exit()

def Manual_drive(keys):
        global Steering
        # 'e' key to start motor
        if keys[pygame.K_e]:  
            print("Starting...")
            odrv0.axis0.requested_state = 8  # Start Motor
            odrv0.axis1.requested_state = 8 # Start Motor

        # 'w' key to move forward
        if keys[pygame.K_w]:
            odrv0.axis1.controller.input_vel = 1 #Stop Wheels before moving forward
            odrv0.axis0.controller.input_vel = -1 
            print ("Forward 1 vel")


        # 's' key to move backwards
        if keys[pygame.K_s]:
            odrv0.axis1.controller.input_vel = -1 #Stop Wheels before moving forward
            odrv0.axis0.controller.input_vel =  1
            print ("Reverse 1 Vel")

        # 'shift' key to brake
        mods = pygame.key.get_mods()  # Get current modifier key states
        if mods & pygame.KMOD_SHIFT:
            odrv0.axis1.controller.input_vel = 0
            odrv0.axis0.controller.input_vel = 0
            print ('STOP')  

        # 'w' key to move forward
        if keys[pygame.K_y]:
            odrv0.axis1.controller.input_vel = 1 #Stop Wheels before moving forward
            print ("Right Wheel 1 vel forward") 


        # 'w' key to move forward
        if keys[pygame.K_h]:
            odrv0.axis1.controller.input_vel = -1 #Stop Wheels before moving forward
            print ("Right Wheel 1 vel reverse") 

        # 'w' key to move forward
        if keys[pygame.K_t]:
            odrv0.axis0.controller.input_vel = -1 #Stop Wheels before moving forward
            print ("Right Wheel 1 vel forward") 


        # 'w' key to move forward
        if keys[pygame.K_g]:
            odrv0.axis0.controller.input_vel = 1 #Stop Wheels before moving forward
            print ("Right Wheel 1 vel reverse") 
        

odrv0 = connect_to_odrive()
last_voltage_print_time = time.time()  # Initialize the timer
while True:
        # Print the ODrive voltage at a 1-second interval
        current_time = time.time()
        if current_time - last_voltage_print_time >= 0.3:  # 1 second interval
            print("ODrive Voltage:", odrv0.vbus_voltage)
            last_voltage_print_time = current_time        # Poll for pygame events
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        Manual_drive(keys)
                # Check for keydown events to detect arrow key presses
        if keys[pygame.K_a]:

            odrv0.axis1.controller.input_vel = 2
            odrv0.axis0.controller.input_vel = -2
            print("Forward 2 Vel")

        elif keys[pygame.K_d]:
            print("Reverse 2 Vel")
            odrv0.axis1.controller.input_vel = -2
            odrv0.axis0.controller.input_vel = 2

        # 'w' key to move forward
        if keys[pygame.K_r]:
            odrv0.axis1.controller.input_vel = 0
            odrv0.axis1.requested_state = 3  # Set ODrive to idle state
            odrv0.axis0.controller.input_vel = 0
            odrv0.axis0.requested_state = 3  # Set ODrive to idle 
            
        # 'w' key to move forward
        elif keys[pygame.K_ESCAPE]:
            odrv0.reboot()
            break
pygame.quit()
