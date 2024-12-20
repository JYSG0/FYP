import pygame
import time

# Initialize pygame
pygame.init()
pygame.joystick.init()

joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]

# Initialize all joysticks
for joystick in joysticks:
    joystick.init()
    print(f"Joystick {joystick.get_name()} initialized.")

# Example: Access joystick[0] if available
if joysticks:
    joystick = joysticks[0]
    print(f"Joystick 0 Name: {joystick.get_name()}")

else:
    print("No joysticks connected.")
while True:
    joystick = joysticks[0]
    joystick.rumble(0, 0.7, 1000)