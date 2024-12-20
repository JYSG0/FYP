import time
import tkinter as tk
import threading
import pygame
# Flags for modes and driving controls
is_manual_mode = True  # Start in manual mode
manual_speed = 0  # Speed simulation (negative for reverse, positive for forward)
manual_turn = 0   # Turn simulation (-1 for left, 1 for right)
auto_thread = None  # To keep track of the autonomous driving thread
auto_running = False  # Flag to indicate whether autonomous driving is running

# Function to simulate autonomous driving
def autonomous_driving(status_label):
    global is_manual_mode, manual_speed, auto_running
    while auto_running:
        if not is_manual_mode:
            manual_speed += 1  # Increment speed automatically
            status_label.config(text=f"Mode: Autonomous Driving\nSpeed: {manual_speed}", fg="green")
            print(f"Autonomous Mode: Speed: {manual_speed}...")
            time.sleep(1)  # Increment speed every second
        else:
            break  # Exit the loop when switching to manual mode

# Function to handle key presses for manual driving
def key_press(keys):
    global manual_speed, manual_turn, is_manual_mode
    if is_manual_mode:
        if keys[pygame.K_w] == 'w':  # Move forward (increment speed)
            manual_speed += 1
            print(f"Manual Mode: Speed: {manual_speed}")
        elif keys[pygame.K_s] == 's':  # Move backward (decrement speed)
            manual_speed -= 1
            print(f"Manual Mode: Speed: {manual_speed}")
        elif keys[pygame.K_a] == 'a':  # Turn left (decrement turn)
            manual_turn -= 1
            print(f"Manual Mode: Turn: {manual_turn}")
        elif keys[pygame.K_d] == 'd':  # Turn right (increment turn)
            manual_turn += 1
            print(f"Manual Mode: Turn: {manual_turn}")
        elif keys[pygame.K_q] == 'q':  # Quit simulation (manual stop)
            print("Simulation stopped by user.")
            root.quit()

# Function to handle mode switching
def toggle_mode(event, status_label):
    global is_manual_mode, auto_thread, auto_running
    if event.char == 'm':  # Switch to autonomous mode
        if not is_manual_mode:  # Already in autonomous mode
            return
        is_manual_mode = False
        auto_running = True  # Start autonomous driving
        status_label.config(text="Mode: Autonomous Driving", fg="green")
        print("Switched to Autonomous Mode")
        # Start autonomous driving thread
        if auto_thread is None or not auto_thread.is_alive():
            auto_thread = threading.Thread(target=autonomous_driving, args=(status_label,))
            auto_thread.daemon = True
            auto_thread.start()

    elif event.char == 'n':  # Switch to manual mode
        if is_manual_mode:  # Already in manual mode
            return
        is_manual_mode = True
        auto_running = False  # Stop autonomous driving
        status_label.config(text="Mode: Manual Driving", fg="blue")
        print("Switched to Manual Mode")

# Function to create and run the UI
def create_ui():
    global root, status_label
    root = tk.Tk()
    root.title("Car Simulation")

    # Create status label to display only the mode
    status_label = tk.Label(root, text="Mode: Manual Driving", font=("Helvetica", 16), fg="blue")
    status_label.pack(pady=20)

    # Quit button
    quit_button = tk.Button(root, text="Quit", font=("Helvetica", 14), command=root.destroy)
    quit_button.pack(pady=10)

    # Bind key events for mode toggling and manual driving control
    root.bind('<KeyPress>', lambda keys: [key_press(keys)])

    root.mainloop()

# Run the UI
create_ui()
    
