from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Dict
import uvicorn
import asyncio
import threading
import json
import os

app = FastAPI()
static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")
templates_directory = os.path.join(os.path.dirname(__file__), "template")
templates = Jinja2Templates(directory=templates_directory)

TCPwriter = None
currentData = {} #Global variable to store latest data sent to socket
currentStep = {}
action = {}
testStep = {}
routeCoords = {
    "startLat": None,
    "startLon": None,
    "endLat": None,
    "endLon": None,
}

instructions = []
waypoints = []
turnAngle = []
modifier = []
turns = []

# Store connected WebSocket clients
connected_clients = []

#WebSocket endpoint for Javascript
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global routeCoords, instructions, waypoints, turnAngle
    await websocket.accept()
    connected_clients.append(websocket)
    print("Websocket client connected")
    try:
        while True:
            # Wait for messages from the WebSocket client
            data = await websocket.receive_text()
            print(f"Received message from WebSocket: {data}")
            
            message = json.loads(data)

            if "startLat" in message:
                routeCoords["startLat"] = message["startLat"]
                routeCoords["startLon"] = message["startLon"]
                routeCoords["endLat"] = message["endLat"]
                routeCoords["endLon"] = message["endLon"]
                print(f"Updated start coordinates: {routeCoords['startLat']}, {routeCoords['startLon']}, {routeCoords['endLat']}, {routeCoords['endLon']}")

            elif "instructions" in message:
                instructions = message["instructions"]  # Extract the instructions
                #Check if FastAPI received
                print(f"Received instructions: {instructions}")

                #Send a confirmation back to the client
                #await websocket.send_json({"status": "instructions received", "instructions": instructions})

            elif "waypoints" in message:
                waypoints = message["waypoints"]  # Extract the waypoints
                #Check if FastAPI received
                print(f"Received waypoints: {waypoints}")

                #Send a confirmation back to the client
                #await websocket.send_json({"status": "waypoints received", "waypoints": waypoints})

            elif "turnAngle" in message:
                turnAngle = message["turnAngle"]  # Extract the instructions
                #Check if FastAPI received
                print(f"Received turnAngles: {turnAngle}")

                #Send a confirmation back to the client
                #await websocket.send_json({"status": "turnAngle received", "turnAngle": turnAngle})

            # Echo the message back to all connected WebSocket clients
            for client in connected_clients:
                await client.send_text(data)
    except Exception as e:
        print(f"WebSocket connection closed: {e}")
    finally:
        connected_clients.remove(websocket)

#TCP server connection for ESP32 to FastAPI
async def receiveFromESP(reader, writer):
    global currentData, TCPwriter
    TCPwriter = writer
    print("ESP32 client connected")
    buffer = ""  # Buffer to accumulate message fragments
    
    while True:
        data = await reader.read(100)  # Read up to 100 bytes
        if not data:
            break  # Exit loop if no data received

        buffer += data.decode()  # Accumulate data in buffer

        try:
            # Attempt to decode JSON once buffer seems complete
            message = json.loads(buffer)
            print(f"Received complete message from ESP32: {message}")
            buffer = ""  # Clear buffer if decoding is successful
            
            # Check if the message is of type "Coords"
            if message.get("type") == "Coords":
                currentData = {
                    "type": "currentCoords",
                    "currentLat": message["latitude"],
                    "currentLon": message["longitude"],
                    "azimuth": message["azimuth"],
                    "direction": message["direction"]
                }
                print(f"Updated currentData: {currentData}")

                # Send a response to ESP32 for checking
                response = "Coords received"
                writer.write(response.encode())
                await writer.drain()

                # Broadcast the updated `currentData` to Javascript
                for client in connected_clients:
                    await client.send_text(json.dumps({"update": currentData}))

                    print(f"Received data: {currentData}")

            # Check if the message is of type "currentStep"
            elif message.get("type") == "action":
                currentStep = {
                    "type": "action",
                    "action": message["action"]
                }
                print(f"Updated step data: {action}")

                # Send a response to ESP32 for checking
                response = "action received"
                writer.write(response.encode())
                await writer.drain()

                # Broadcast the updated `currentData` to Javascript
                for client in connected_clients:
                    await client.send_text(json.dumps({"update": currentStep}))

                    print(f"Received data: {currentData}")

            elif message.get("type") == "testStep":
                testStep = {
                    "type": "testStep",
                    "currentStep": message["testStep"]
                }
                print(f"Updated step data: {testStep}")

                # Send a response to ESP32 for checking
                response = "testStep received"
                writer.write(response.encode())
                await writer.drain()

                # Broadcast the updated `currentData` to Javascript
                for client in connected_clients:
                    await client.send_text(json.dumps({"update": testStep}))

                    print(f"Received data: {testStep}")

        except json.JSONDecodeError:
            print("Error decoding JSON from esp32")

    print("Closing ESP32 connection")
    writer.close()
    await writer.wait_closed()

async def startTCPServer():
    server = await asyncio.start_server(receiveFromESP, "0.0.0.0", 8765)    #Same as ESP32
    async with server:
        await server.serve_forever()

def runTCPServer():
    asyncio.run(startTCPServer())

#HTTP for FastAPI to ESP32
#Link to HTML file
@app.get("/", response_class=HTMLResponse)
async def map(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/get-routeData")
async def getRouteDataESP32():
#Endpoint for ESP32 to get waypoints and instructions in JSON format
    return JSONResponse(content=routeCoords)

@app.get("/get-navData", response_model=Dict[str, List])
async def getNavDataESP32():
    #Endpoint for ESP32 to get waypoints and instructions in JSON format
    data = {
        "waypoints": waypoints,
        "instructions": instructions,
        "turnAngle": turnAngle
    }
    print(data)
    return data

if __name__ == "__main__":
    # Start the TCP server in a separate thread
    tcp_thread = threading.Thread(target=runTCPServer, daemon=True)
    tcp_thread.start()

    # Start the FastAPI server with Uvicorn
    uvicorn.run(app, host="192.168.167.96", port=5501) #Same as JS
