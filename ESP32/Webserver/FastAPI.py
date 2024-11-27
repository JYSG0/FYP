#Importing the 
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
 #Global variable to store latest data sent to socket
currentStep = {}
action = {}
routeActive = {}
testStep = {}
routeCoords = {
    "startLat": None,
    "startLon": None,
    "endLat": None,
    "endLon": None,
}

currentCoords = {
    "currentLat": None,
    "currentLon": None,
    "azimuth": None,
    "direction": None
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
    global routeCoords, instructions, waypoints, turnAngle, routeActive
    await websocket.accept()
    connected_clients.append(websocket)
    print("Websocket client connected")
    try:
        while True:
            # Wait for messages from the WebSocket client
            data = await websocket.receive_text()
            print(f"Received message from WebSocket: {data}")
            
            #Decode outer JSON string
            firstMessage = json.loads(data)
            print(firstMessage)
            print(f"Type of data: {type(firstMessage)}")

            if isinstance(firstMessage, str):
                print("Parsed message is not a dictionary")
                message = json.loads(firstMessage)
                print(message)
                print(f"Type of data: {type(message)}")
            else:
                message = firstMessage
                print(message)

            if "latitude" in message:
                print("latitude")
                currentCoords["currentLat"] = message["latitude"]
                currentCoords["currentLon"] = message["longitude"]
                currentCoords["azimuth"] = message["azimuth"]
                currentCoords["direction"] = message["direction"]
                print(f"Updated current coordinates: {currentCoords['currentLat']}, {currentCoords['currentLon']}, {currentCoords['azimuth']}, {currentCoords['direction']}")

            elif "startLat" in message:
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

                #Send a confirmation back to the client for debugging
                #await websocket.send_json({"status": "waypoints received", "waypoints": waypoints})

            elif "turnAngle" in message:
                turnAngle = message["turnAngle"]  # Extract the instructions
                #Check if FastAPI received
                print(f"Received turnAngles: {turnAngle}")

                #Send a confirmation back to the client
                #await websocket.send_json({"status": "turnAngle received", "turnAngle": turnAngle})

            elif "routeActive" in message:
                routeActive = message["routeActive"]  # Extract the instructions
                #Check if FastAPI received
                print(f"Received routeActive: {routeActive}")

                #Send a confirmation back to the client
                #await websocket.send_json({"status": "turnAngle received", "turnAngle": turnAngle})
            else:
                print("Unknown parsed message: ", message)

            # Echo the message back to all connected WebSocket clients - Must be a JSON string
            for client in connected_clients:
                await client.send_text(firstMessage)
    except Exception as e:
        print(f"WebSocket connection closed: {e}")
    finally:
        connected_clients.remove(websocket)

#HTTP for FastAPI to ESP32
#Link to HTML file
@app.get("/", response_class=HTMLResponse)
async def map(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

#Endpoint for ESP32 to get start and end coordinates from routeCoords
@app.get("/get-routeData")
async def getRouteDataESP32():
    return JSONResponse(content=routeCoords)    #FastAPI sends it in JSON formate

@app.get("/get-routeActive")
async def getRouteActiveESp32():
    data = {
        "routeActive": routeActive
    }
    print(data)
    return JSONResponse(content=data)

#Endpoint for ESP32 to get waypoints and instructions and turnAngle
@app.get("/get-navData", response_model=Dict[str, List])
async def getNavDataESP32():
    data = {
        "waypoints": waypoints,
        "instructions": instructions,
        "turnAngle": turnAngle
    }
    print(data)
    return data

if __name__ == "__main__":
    # Start the FastAPI server with Uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5501) #Same as JS
