import asyncio
from bleak import BleakClient, BleakError
import json
import websockets

# Replace this with your ESP32's MAC address
DEVICE_ADDRESS = "88:13:BF:6F:E0:B6"

SERVICE_UUID = "19b10000-e8f2-537e-4f6c-d104768a1214"
SENSOR_CHAR_UUID = "19b10001-e8f2-537e-4f6c-d104768a1214"  # ESP32 sends to this UUID
LED_UUID = "19b10002-e8f2-537e-4f6c-d104768a1214"          # ESP32 receives from this UUID

WEBSOCKET_URL_FASTAPI = "ws://127.0.0.1:5501/ws"
WEBSOCKET_URL_JETSON = "ws://127.0.0.1:8765/ws"


async def list_services(client):
    """Lists available BLE services and characteristics."""
    print("Listing services and characteristics:")
    for service in client.services:
        print(f"Service: {service.uuid}")
        for char in service.characteristics:
            print(f"  Characteristic: {char.uuid}, Properties: {char.properties}")


async def read_sensor_data(client):
    """Reads sensor data from BLE device."""
    try:
        sensor_data = await client.read_gatt_char(SENSOR_CHAR_UUID)
        data = sensor_data.decode("utf-8")
        print(f"Sensor Data: {data}")
        return data
    except Exception as e:
        print(f"Error reading sensor data: {e}")
        raise  # Propagate for reconnection


async def send_sensor_data(client, json_data):
    """Sends JSON data to the BLE device in chunks."""
    try:
        json_bytes = json_data.encode("utf-8")
        chunk_size = 20
        total_chunks = len(json_bytes) // chunk_size + (1 if len(json_bytes) % chunk_size else 0)

        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = json_bytes[start_idx:end_idx]
            is_last_chunk = 1 if i == total_chunks - 1 else 0
            chunk_with_metadata = bytes([is_last_chunk]) + chunk
            await client.write_gatt_char(LED_UUID, chunk_with_metadata)
            print(f"Sent chunk {i + 1}/{total_chunks}")
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"Error sending data: {e}")
        raise


async def read_data(client, websocketFastAPI, websocketJetson):
    """Reads BLE data and forwards it to WebSocket."""
    try:
        while True:
            sensor_data = await read_sensor_data(client)
            if sensor_data:
                await websocketFastAPI.send(json.dumps({"sensor_value": sensor_data}))
                await websocketJetson.send(json.dumps({"sensor_value": sensor_data}))
                print("Sent sensor data to WebSockets")
            await asyncio.sleep(1)  # Adjust interval as needed
    except Exception as e:
        print(f"Error reading BLE data: {e}")
        raise


async def process_websocket_messages(websocket, client):
    """Processes incoming WebSocket messages and sends them to BLE."""
    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                print(f"Received WebSocket message: {message}")

                # Forward valid data to BLE
                JSONdata = json.loads(message)
                if "latitude" not in JSONdata and "longitude" not in JSONdata:
                    await send_sensor_data(client, json.dumps(JSONdata))

            except asyncio.TimeoutError:
                continue  # No message; continue listening
    except Exception as e:
        print(f"Error processing WebSocket messages: {e}")
        raise


async def main():
    """Main loop managing BLE and WebSocket connections."""
    while True:
        try:
            async with BleakClient(DEVICE_ADDRESS) as client:
                print(f"Connected to BLE device: {DEVICE_ADDRESS}")
                await list_services(client)

                async with websockets.connect(WEBSOCKET_URL_FASTAPI) as websocketFastAPI, \
                        websockets.connect(WEBSOCKET_URL_JETSON) as websocketJetson:
                    await asyncio.gather(
                        read_data(client, websocketFastAPI, websocketJetson),
                        process_websocket_messages(websocketFastAPI, client)
                    )
        except (BleakError, websockets.ConnectionClosedError) as e:
            print(f"Connection error: {e}. Reconnecting...")
            await asyncio.sleep(1)
        except Exception as e:
            print(f"Unexpected error: {e}. Reconnecting...")
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
