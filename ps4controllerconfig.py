import pygame
import time

# Initialize Pygame
pygame.init()

# Initialize the joystick module
pygame.joystick.init()

# Check if the PS4 controller is connected
joystick_count = pygame.joystick.get_count()
if joystick_count == 0:
    print("No controller detected. Please connect your PS4 controller.")
else:
    print(f"{joystick_count} controller(s) detected.")

# Select the first controller
joystick = pygame.joystick.Joystick(0)
joystick.init()

# PS4 controller trigger axes:
# Typically, L2 (left trigger) is axis 4 and R2 (right trigger) is axis 5.
left_trigger_axis = 4
right_trigger_axis = 5

# Joystick axis threshold for movement sensitivity
joystick_movement_threshold = 0.1

# Track trigger states
right_trigger_pressed = False
left_trigger_pressed = False

# Throttle percentage (for R2 trigger control)
throttle_percentage = 0

# PS4 controller button mappings for right-side buttons:
# Cross (X) = Button 0
# Circle (O) = Button 1
# Triangle (△) = Button 2
# Square (□) = Button 3
button_mapping = {
    0: "Cross (X)",
    1: "Circle (O)",
    2: "Square (□)",
    3: "Triangle (△)"
}

# Track button states (to know if a button is held down)
buttons_pressed = {0: False, 1: False, 2: False, 3: False}

# Joystick axes (0 = left stick horizontal, 1 = left stick vertical, etc.)
horizontal_axis = 0  # Left stick horizontal (left/right)
vertical_axis = 1    # Left stick vertical (up/down)

# Loop to get controller events
while True:
    for event in pygame.event.get():
        # Handle joystick and trigger events
        if event.type == pygame.JOYAXISMOTION:
            # Check the left trigger (L2) axis
            if event.axis == left_trigger_axis:
                if event.value > 0.1:  # Trigger is pressed beyond a small threshold
                    left_trigger_pressed = True
                else:
                    left_trigger_pressed = False

            # Check the right trigger (R2) axis for throttle control
            elif event.axis == right_trigger_axis:
                trigger_value = event.value  # Value between 0.0 (not pressed) and 1.0 (fully pressed)
                if trigger_value > 0.1:  # Significant trigger press for throttle
                    throttle_percentage = int(trigger_value * 100)  # Scale to 0-100%
                    right_trigger_pressed = True
                else:
                    throttle_percentage = 0  # No throttle when trigger is not pressed
                    right_trigger_pressed = False

            # Handle left stick horizontal movement (left/right)
            if event.axis == horizontal_axis:
                axis_value = event.value
                if abs(axis_value) > joystick_movement_threshold:  # Significant movement threshold
                    degree_of_movement = int(abs(axis_value) * 100)  # Scale movement (0 to 100%)
                    if axis_value < 0:
                        print(f"Moving left, {degree_of_movement}%")
                    elif axis_value > 0:
                        print(f"Moving right, {degree_of_movement}%")

            # Handle left stick vertical movement (up/down)
            elif event.axis == vertical_axis:
                axis_value = event.value
                if abs(axis_value) > joystick_movement_threshold:  # Significant movement threshold
                    degree_of_movement = int(abs(axis_value) * 100)  # Scale movement (0 to 100%)
                    if axis_value < 0:
                        print(f"Moving up, {degree_of_movement}%")
                    elif axis_value > 0:
                        print(f"Moving down, {degree_of_movement}%")

        # Handle button press events for right-side buttons
        if event.type == pygame.JOYBUTTONDOWN:
            if event.button in button_mapping:
                buttons_pressed[event.button] = True

        if event.type == pygame.JOYBUTTONUP:
            if event.button in button_mapping:
                buttons_pressed[event.button] = False

    # Continuously print messages based on the trigger states
    if right_trigger_pressed:
        print(f"Accelerating with throttle: {throttle_percentage}%")
    if left_trigger_pressed:
        print("Brake")

    # Continuously print buttons that are being held down
    for button, is_pressed in buttons_pressed.items():
        if is_pressed:
            print(f"{button_mapping[button]} button pressed")

    # Limit the loop to a reasonable refresh rate
    time.sleep(0.1)
