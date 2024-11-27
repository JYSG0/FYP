import asyncio
from bleak import BleakClient
import json
import websockets

# Replace this with your ESP32's MAC address
DEVICE_ADDRESS = "88:13:BF:6F:E0:B6"

SERVICE_UUID = "19b10000-e8f2-537e-4f6c-d104768a1214"
SENSOR_CHAR_UUID = "19b10001-e8f2-537e-4f6c-d104768a1214"
LED_UUID = '19b10002-e8f2-537e-4f6c-d104768a1214'

WEBSOCKET_URL = "ws://127.0.0.1:5501/ws"

async def list_services(client):
    print("Listing services and characteristics:")
    for service in client.services:
        print(f"Service: {service.uuid}")
        for char in service.characteristics:
            print(f"  Characteristic: {char.uuid}, Properties: {char.properties}")

async def read_sensor_data(client):
    try:
        data = await client.read_gatt_char(SENSOR_CHAR_UUID)
        if data:
            value = data.decode("utf-8")
            jsondata = json.loads(value)
            print(f"Received sensor data: {jsondata}")
            return jsondata
    except Exception as e:
        print(f"Error reading sensor data: {e}")
    return None

async def send_sensor_data(client, message):
    try:
        data = message.encode("utf-8")
        print(f"Payload size: {len(message)} bytes")
        print(data)
        await client.write_gatt_char(LED_UUID, data)
        print(f"Sent to ESP32: {message}")
    except Exception as e:
        print(f"Error sending to ESP32: {e}")

async def main():
    async with BleakClient(DEVICE_ADDRESS) as client:
        print(f"Connected to {DEVICE_ADDRESS}")
        await list_services(client)

        async with websockets.connect(WEBSOCKET_URL) as websocket:
            print("Connected to WebSocket")
            try:
                while True:
                    sensor_data = await read_sensor_data(client)
                    if sensor_data:
                        await websocket.send(json.dumps(sensor_data))
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                        print(f"Received from WebSocket: {response}")

                        # if "startLat" in response or "attributes" in response or "routeActive" in response:
                        await send_sensor_data(client, response)
                    except asyncio.TimeoutError:
                        pass  # No message received; continue loop
                    
                    await asyncio.sleep(0.5)
            except (asyncio.CancelledError, KeyboardInterrupt):
                print("Stopped by user.")
            except Exception as e:
                print(f"Error in main loop: {e}")

asyncio.run(main())
