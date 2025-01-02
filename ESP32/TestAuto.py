import asyncio

# Global state
auto_mode = False
msg = None
key = None

async def lidar():
    print("lidar")
    await asyncio.sleep(0.1)  # Simulated delay

async def toggle():  # Read key and handle switching between auto and manual
    global auto_mode, key
    key = "m"  # Simulate a key press for testing
    if key == "m":
        auto_mode = not auto_mode
        print(f"Switched to {'Auto' if auto_mode else 'Manual'} mode")

async def controls():  # Keyboard controls
    global auto_mode, key
    key = "w"
    print("controls")
    if auto_mode:
        print("Auto mode")
        await auto_control()
    else:
        print("Manual mode")
        await manual_control()

async def auto_control():
    global msg
    if msg == "straight":
        print("Wheels front")
    elif msg == "left":
        print("Turn left")
    elif msg == "right":
        print("Turn right")
    elif msg == "sharp left":
        print("Sharp left")
    elif msg == "sharp right":
        print("Sharp right")
    else:
        print("Unknown message")
    await asyncio.sleep(0.1)  # Simulated delay

async def manual_control():
    global key
    if key == "w":
        print("Wheels front")
    elif key == "a":
        print("Turn left")
    elif key == "d":
        print("Turn right")
    else:
        print("Unknown message")
    await asyncio.sleep(0.1)  # Simulated delay

async def detect():  # Object detection
    print("detect")
    await asyncio.sleep(0.1)  # Simulated delay

async def read():  # Read from ESP32
    global msg
    # Simulated reading logic (replace with actual ESP32 communication)
    msg = "straight"
    print("read message")
    await asyncio.sleep(0.1)

async def main():
    while True:
        await asyncio.gather(
            lidar(),
            toggle(),
            controls(),
            detect(),
            read()
        )
        await asyncio.sleep(0.1)  # Prevent event loop starvation

if __name__ == "__main__":
    asyncio.run(main())
