#DEVICE_ADDRESS = "88:13:BF:6F:E0:B6"
import asyncio
from bleak import BleakClient
import json
import websockets

# Replace this with your ESP32's MAC address
DEVICE_ADDRESS = "88:13:BF:6F:E0:B6"

SERVICE_UUID = "19b10000-e8f2-537e-4f6c-d104768a1214"
SENSOR_CHAR_UUID = "19b10001-e8f2-537e-4f6c-d104768a1214"
LED_UUID = '19b10002-e8f2-537e-4f6c-d104768a1214'


WEBSOCKET_URL_FASTAPI = "ws://127.0.0.1:5501/ws"
WEBSOCKET_URL_JETSON = "ws://127.0.0.1:8765/ws"

frontend = {
    "currentStep": None,
    "distanceToTurn": None,
}

msg = {}

async def list_services(client):
    print("Listing services and characteristics:")
    for service in client.services:
        print(f"Service: {service.uuid}")
        for char in service.characteristics:
            print(f"  Characteristic: {char.uuid}, Properties: {char.properties}")

async def read_sensor_data(client):
    try:
        sensor_data = await client.read_gatt_char(SENSOR_CHAR_UUID)
        print(sensor_data)
        data = sensor_data.decode('utf-8')
        print(f"Sensor Data: {data}")
        return data
    except Exception as e:
        print(f"Error reading sensor data: {e}")

async def read_data(client, websocketFastAPI, websocketJetson):
    """Read data from BLE characteristics and send to WebSocket."""
    try:
        while True:
            sensor_data = await read_sensor_data(client)
            # if "msg" in sensor_data:
            #     websocket = websocketJetson
            # else:
            #     print("FastAPI")
            #     websocket = websocketFastAPI
            print(sensor_data)
            if "msg" in sensor_data:
                await websocketFastAPI.send(json.dumps(sensor_data))
                await websocketJetson.send(json.dumps(sensor_data))
            else: 
                await websocketFastAPI.send(json.dumps(sensor_data))
            print("sent")

    except Exception as e:
        print(f"Error reading data: {e}")

async def process_websocket_messages(websocket, client):
    """Handle WebSocket messages and send data via BLE."""
    try:
        while True:
            try:
                # Receive from WebSocket
                response = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                print(f"Received from WebSocket: {response}")
                
                # Parse JSON data
                JSONdata = json.loads(response)
                print(JSONdata)

                # Check if 'latitude', 'longitude', or 'msg' exists in the message
                if "latitude" in JSONdata or "longitude" in JSONdata or "msg" in JSONdata:
                    print("Message contains 'latitude', 'longitude', or 'msg'; skipping data echo.")
                else:
                    # Convert the data to JSON string
                    JSONdata = json.dumps(JSONdata)
                    print("Sending large JSON data in chunks...")
                    await send_sensor_data(client, JSONdata)

            except asyncio.TimeoutError:
                pass  # No message received; continue loop

    except Exception as e:
        print(f"Error processing WebSocket messages: {e}")

async def send_sensor_data(client, json_data):
    try:
        json_bytes = json_data.encode('utf-8')  # Convert JSON to bytes
        chunk_size = 20  # Maximum number of bytes per chunk (BLE limit)
        total_chunks = len(json_bytes) // chunk_size + (1 if len(json_bytes) % chunk_size else 0)
        
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = json_bytes[start_idx:end_idx]
            
            # Indicate if it's the last chunk
            is_last_chunk = 1 if i == total_chunks - 1 else 0
            
            # Prepare the chunk with metadata (start/end indicator)
            chunk_with_metadata = bytes([is_last_chunk]) + chunk
            
            # Send the chunk
            await client.write_gatt_char(LED_UUID, chunk_with_metadata)
            print(f"Sent chunk {i + 1}/{total_chunks}")
            
            await asyncio.sleep(0.1)  # Small delay between chunks

    except Exception as e:
        print(f"Error sending JSON in chunks: {e}")

async def main():
    async with BleakClient(DEVICE_ADDRESS) as client:
        print(f"Connected to {DEVICE_ADDRESS}")
        await list_services(client)

        async with websockets.connect(WEBSOCKET_URL_FASTAPI) as websocketFastAPI, \
                   websockets.connect(WEBSOCKET_URL_JETSON) as websocketJetson:
            try:
                # Run read_data and process_websocket_messages concurrently
                await asyncio.gather(
                    read_data(client, websocketFastAPI, websocketJetson),
                    process_websocket_messages(websocketFastAPI, client)
                )
            except (asyncio.CancelledError, KeyboardInterrupt):
                print("Stopped by user.")
            except Exception as e:
                print(f"Error in main loop: {e}")

asyncio.run(main())
