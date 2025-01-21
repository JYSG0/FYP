import time
import math
import re
import machine
import asyncio
import aioble
import bluetooth
from micropyGPS import MicropyGPS
import ujson
import json
import network
from bluetooth import BLE

#from micropython import const
from machine import I2C, Pin
from hmc5883L import HMC5883L	#Library for QMC5883L compass
import struct

ble = BLE()
ble.active(True)
ble.config(mtu=256)

_BLE_SERVICE_UUID = bluetooth.UUID('19b10000-e8f2-537e-4f6c-d104768a1214')
_BLE_SENSOR_CHAR_UUID = bluetooth.UUID('19b10001-e8f2-537e-4f6c-d104768a1214')  #GPS Data UUID
_BLE_WRITE_UUID = bluetooth.UUID('19b10002-e8f2-537e-4f6c-d104768a1214')

_ADV_INTERVAL_MS = 250_000

#Register GATT service and characteristics
ble_service = aioble.Service(_BLE_SERVICE_UUID)
sensor_characteristic = aioble.Characteristic(ble_service, _BLE_SENSOR_CHAR_UUID, read=True, notify=True)
write_characteristic = aioble.Characteristic(ble_service, _BLE_WRITE_UUID, read=True, write=True, notify=True, capture=True)

# Initialize the station (WiFi) interface
sta_if = network.WLAN(network.STA_IF)

# Activate the station interface
sta_if.active(True)

# Get the MAC address
mac_address = sta_if.config('mac')
formatted_mac = ':'.join(f'{b:02X}' for b in mac_address)

#Register services
aioble.register_services(ble_service)

#Instantiate the micropyGPS object
my_gps = MicropyGPS()

#Define the UART pins and create a UART object
gps_serial = machine.UART(2, baudrate=9600, tx=17, rx=16)

#Define the SCL and SDA pins
compass = HMC5883L(scl=22, sda=21)

#Variables
instructions = []
turnProperties = []
modifiers = []
turns = []
bearingAfter = []
bearingBefore = []
step = 0    #Compare turning directions
currentStep = 0

#Boolean checks
matchedBearing = False
reachedDestination = False
routeActive = False
previousRouteActive = False

data_buffer = bytearray()

compass_data = {}
gps_data = {}
dataToSend = {}

#Calibrate compass
#Function to read compass values
async def readCompass():
    global azimuth, direction, compass_data
    x, y, z = compass.read()
    azimuth, direction = compass.format_result(x, y, z)
    
    #Send data to webserver
    compass_data = {
        "azimuth": azimuth,
        "direction": direction
    }

async def sendData(data):
    dataToSend = ujson.dumps(data).encode('utf-8')
    print("Sending combined data: ", dataToSend)
    sensor_characteristic.write(dataToSend, send_update=True)
    await asyncio.sleep_ms(500)

#Function to read GPS values
async def readGPS():
    global latitude, longitude, gps_data
    
    while gps_serial.any():
        data = gps_serial.read()	#Get GPS data
        for byte in data:
            stat = my_gps.update(chr(byte))
            if stat is not None:
                latitude_str = my_gps.latitude_string()
                longitude_str = my_gps.longitude_string()

                #Parse latitude
                lat_deg = int(latitude_str.split('°')[0])
                lat_min = float(latitude_str.split('°')[1].split("'")[0])
                lat_dir = latitude_str[-1]  #Get direction (N/S)

                #Parse longitude
                lon_deg = int(longitude_str.split('°')[0])
                lon_min = float(longitude_str.split('°')[1].split("'")[0])
                lon_dir = longitude_str[-1]  #Get direction (E/W)

                #Convert to decimal degrees
                latitude = dmm_to_dd(lat_deg, lat_min, lat_dir)
                longitude = dmm_to_dd(lon_deg, lon_min, lon_dir)

                #Get precision performance
                #satellites = my_gps.satellites_in_use()
                #precision = my_gps.hdop()
                
                #print('lat:', latitude)
                #print('lon:', longitude)
                # print('satellites: ', satellites)
                # print('precision: ', precision)
                
                #Send data to web server
                gps_data = {
                    "latitude": latitude,
                    "longitude": longitude,
                }
# async def readGPS():
#     global latitude, longitude, gps_data
#     latitude = 1.3088
#     longitude = 103.7766
#     #Send data to web server
#     gps_data = {
#         "latitude": latitude,
#         "longitude": longitude,
#     }
                
def convert_to_sgt(utc_time):
    hour, minute, second = utc_time
    hour += 8  #Adjust for Singapore Time (SGT)
    if hour >= 24:
        hour -= 24
    return hour, minute, second

def dmm_to_dd(degrees, minutes, direction):
    decimal_degrees = degrees + (minutes / 60)
    if direction in ['S', 'W']:  #Southern and Western directions are negative
        decimal_degrees = -decimal_degrees
    return decimal_degrees

#Reset start, destination coords
def reset_coordinates():
    global startLat, startLon, endLat, endLon  #Ensure to declare them as global
    startLat = 0.0  #or your desired initial value
    startLon = 0.0  #or your desired initial value
    endLat = 0.0    #or your desired initial value
    endLon = 0.0    #or your desired initial value
    print(f"Coordinates reset to Start: ({startLat}, {startLon}), Destination: ({endLat}, {endLon})")

#Calculate bearing and distance between 2 points on Earth
def haversine_Bearing(lat1, lon1, lat2, lon2): 
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    
    initial_bearing = math.atan2(x, y)
    compass_bearing = (math.degrees(initial_bearing) + 360) % 360
    
    R = 6372.8  #Radius of Earth in kilometers
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(lat1) * math.cos(lat2) * 
         math.sin(dlon / 2) ** 2)
    distance = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)) * 1000  #Distance in meters

    return compass_bearing, distance

#Instructions sequence for vehicle on route
async def followInstructions(modifiers, turns, bearingAfter, bearingBefore, turningPoints, currentLat, currentLon, threshold=0.8):  #Check back to see if need bearingWLocations
    global azimuth, currentStep, reachedDestination, dataToSend
    tolerance = 10	#+- of angle to turn
    reachedDestination = False
    
    print('currentStep: ', currentStep)
    #Fetch the current turning point based on currentStep
    print('turningPoints[currentStep]: ', turningPoints[currentStep])
    turnLon, turnLat = turningPoints[currentStep]
    print(f'turnLon: {turnLon}, turnLat: {turnLat}')
    target_bearing = bearingAfter[currentStep]
    maxTargetBearing = (target_bearing + tolerance) % 360
    minTargetBearing = (target_bearing - tolerance + 360) % 360

    if minTargetBearing < maxTargetBearing:  #Standard range
        within_tolerance = minTargetBearing <= azimuth <= maxTargetBearing
    else:  #Wrap-around range
        within_tolerance = azimuth >= minTargetBearing or azimuth <= maxTargetBearing

    #Print current coordinates and the target turning point
    print(f"Current Location: Latitude: {currentLat}, Longitude: {currentLon}")
    print(f"Next Turning Point: Latitude: {turnLat}, Longitude: {turnLon}")
    
    #Calculate the distance to the current turning point
    _, distance_to_turn = haversine_Bearing(currentLat, currentLon, turnLat, turnLon)

    print("")
    print(distance_to_turn)	#debugging line
    print(modifiers)
    print(turns)
    turn = turns[currentStep]
    
    if distance_to_turn <= threshold:
        print("In threshold")
        action = f"Action (Time to turn!)"
        #Check azimuth to target_bearing
        #angleToTurn = abs((bearingAfter[currentStep] - azimuth + 180) % 360 - 180)   #positive is left, negative is right
        modifier = modifiers[currentStep]

        #If the turnType is a turn
        if 'turn' in turn:
                #angle = bearingBefore[currentStep] - target_bearing #Set
                angle = (bearingBefore[currentStep] - target_bearing + 180) % 360 - 180
                
                msg = f"Turn {modifier} {abs(angle):.2f} degrees."

                if within_tolerance:
                    msg += " Stop turning, angle reached."
                    currentStep += 1  #Proceed to the next step
                    jetson = "straight"
                else:
                    msg += " Keep turning."
                    jetson = f"Turn {angle}"
            #action = f"{action} for {angle} degrees, {turnOri}"
        
        elif 'depart' in turn:
            msg = "Depart."

            if within_tolerance:
                msg += " Stop turning, angle reached."
                angle = 0
                currentStep += 1  #Proceed to the next step
                jetson = "straight"
            else:
                msg += " Keep turning."
                angle = (azimuth - target_bearing + 180) % 360 - 180
                jetson = f"Turn {angle}"
                

        elif 'arrive' in turn:
            msg = "Arrive"
            reachedDestination = True

            print("Destination reached, stop vehicle")

            if within_tolerance:
                msg += " Stop turning, angle reached."
                angle = 0
                jetson = "Stop"
            else:
                msg += " Keep turning."
                angle = (azimuth - target_bearing + 180) % 360 - 180
                jetson = f"Turn {angle}"

        print(action)
        #await sensor_characteristic.write(msg)
        print(currentStep)

    else:
        #Continue moving towards the turning point
        msg = f"Keep going straight. Distance to next turn: {distance_to_turn:.2f} meters."
        jetson = "straight"
        time.sleep(1)  #Simulate movement delay
        angle = 0
        modifier = None
        #angleToTurn = 0
        within_tolerance = False
    
    print(msg)
    
    dataToSend = {
        "type": "vehicleControl",
        "msg": msg,
        "currentStep": currentStep,
        "distanceToTurn": distance_to_turn,
        "modifier": modifier,
        "angleToTurn": angle,
        "within_tolerance": within_tolerance,
        "jetson": jetson
    }
        
    await sendData(dataToSend)

#Prepare to start main - functions that run before the main but only once
async def prepare():
    global latitude, longitude, azimuth, direction, startLat, startLon, endLat, endLon, lat, lon
    global currentStep, reachedDestination, modifiers, bearingAfter, turns, step, matchedBearing, routeActive, previousRouteActive
    step = 0
    matchedBearing = False  #Initialise it to False at the start
    routeActive = False	#Initialise it to False at the start
    previousRouteActive = True
    
    #Run the main task
    reset_coordinates()
    
#Function to decode and process the received JSON data

#Function to assign data to their respective variables
def assignData(data):
    global modifiers, bearingAfter, bearingBefore, turningPoints, routeActive, turns
    print("asssign data")
    
    if data["type"] == 'modifier':
        modifiers = data['modifier']
        print("modifier: ", modifiers)
    elif data["type"] == 'w':
        turningPoints = data['waypoints']
        print("turningPoints: ",turningPoints)
    elif data["type"] == 'bearingBefore':
        bearingBefore = data['bearingBefore']
        print("bearingBefore: ",bearingBefore)
    elif data["type"] == 'bearingAfter':
        bearingAfter = data['bearingAfter']
        print("bearingAfter: ",bearingAfter)
    elif data["type"] == 'turnTypes':
        turns = data['turnTypes']
        print("turns: ",turns)
    elif data["type"] == 'routeActive':
        routeActive = data['routeActive']
        print("routeActive: ",routeActive)
    else:
        print(f"Unable to idenitfy data type: ", data["type"])
        
def sanitize_json(data):
    try:
        print("Sanitizing data...")
        print("Original Data:", data)

        # Step 1: Convert dictionary input to JSON string
        if isinstance(data, dict):
            print("Input is a dictionary. Converting to JSON string.")
            data = json.dumps(data)
            # Step 3: Replace known incorrect keys and values
            fixed_data = data
            replacements = {
            '"routeActie"': '"routeActive"',
            'bearingBeore': '"bearingBefore"',
            'bearingAfer': '"bearingAfter"',
            '"waypints":': '"waypoints":',
            'deart': '"depart"',
            'arrie': '"arrive"',
            'lft': '"left"',
            'rght': '"right"',
            'rue': 'true',
            'alse': 'false',
            'True': 'true',
            'False': 'false',
            ', .': ', 1.',  # Missing longitude fix
            ', [13.': ', [103.',  # Incorrect prefix fix
            'nll': 'null'
            }

            for old, new in replacements.items():
                fixed_data = fixed_data.replace(old, new)

            print("Fixed Data:", fixed_data)

        # Step 2: Ensure the input is a string
        if isinstance(data, str):
            print("Input is a string")
            
            # Step 2: Replace known incorrect keys and values
            fixed_data = data
            fixed_data = fixed_data.replace('"routeActie"', '"routeActive"')
            fixed_data = fixed_data.replace('"bearingBeore"', '"bearingBefore"')
            fixed_data = fixed_data.replace('"bearingBefor"', '"bearingBefore"')
            fixed_data = fixed_data.replace('"bearingAfer"', '"bearingAfter"')
            fixed_data = fixed_data.replace('"waypints":', '"waypoints":')
            fixed_data = fixed_data.replace('"deart"', '"depart"')
            fixed_data = fixed_data.replace('"turn,', '"turn",')
            fixed_data = fixed_data.replace('"arrie"', '"arrive"')
            fixed_data = fixed_data.replace('"lft"', '"left"')
            fixed_data = fixed_data.replace('"rght"', '"right"')
            fixed_data = fixed_data.replace('rue', 'true').replace('alse', 'false')
            fixed_data = fixed_data.replace('True', 'true').replace('False', 'false')
            fixed_data = fixed_data.replace('"turnTypes, ', '"turnTypes", ')

            # Step 3: Fix specific data issues
            fixed_data = fixed_data.replace(', .', ', 1.')  # Missing longitude fix
            fixed_data = fixed_data.replace(', [13.', ', [103.')  # Incorrect prefix fix
            fixed_data = fixed_data.replace(', 113}', ', 113]}') 
            fixed_data = fixed_data.replace('nll', 'null')  # Fix typos like `nll`
            
            print(fixed_data)

        # Step 4: Validate and parse JSON
        try:
            json_data = ujson.loads(fixed_data)  # Parse into a dictionary
            print("Parsed JSON Data:", json_data)
            assignData(json_data)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            raise ValueError("Invalid JSON format after sanitization.")

        print("Type of json_data:", type(json_data))
        # Step 5: Re-encode and return the sanitized JSON
        #sanitized_json = json.dumps(json_data, indent=2)  # Beautify output
        #print("Sanitized JSON:", sanitized_json)
        return json_data

    except Exception as e:
        print(f"Error sanitizing JSON: {e}")
        return None

# Task to handle incoming writes (handling fragmentation)
async def wait_for_write():
    global data_buffer
    while True:
        try:
            #Wait for data to be written to the characteristic
            connection, data = await write_characteristic.written()
            
            if data:
                is_last_chunk, chunk = struct.unpack("B", data[0:1])[0], data[1:]
                data_buffer.extend(chunk)  #Add the chunk to the buffer

                #print(f"Received chunk, is_last_chunk={is_last_chunk}")
                
                #Check if it's the last chunk and if we've received the full message
                if is_last_chunk:
                    #Reassemble the full message
                    full_data = data_buffer.decode('utf-8')
                    print(f"Full data received: {full_data}")
                    
                    formattedJSON = sanitize_json(full_data)
                    #Process the received JSON data
                                        
                    #Reset the buffer for future data
                    data_buffer = bytearray()
                else:
                    print("Waiting for more chunks...")
        except Exception as e:
            print(f"Error receiving data: {e}")

#Main loop
async def main(connection):
    while connection is not None:
        if connection.is_connected():
            global latitude, longitude, azimuth, direction, startLat, startLon, endLat, endLon
            global modifiers, bearingAfter, bearingBefore, turns, routeActive, previousRouteActive, turningPoints
            global currentStep, reachedDestination, step, matchedBearing, dataToSend
            try:
                
                print(routeActive)
                if not reachedDestination:
                    print('received')
                    
                #Inside the while connection is not None loop
                    await asyncio.sleep(0.5)  #Yield control to the event loop
                    
                    #Check for route stop
                    print("RouteActive: ", routeActive)
                    if not routeActive and previousRouteActive:  #routeActive was True, now it's False
                        print("Route stopped. Resetting instructions and turning points.")
                        reset_coordinates()
                        turningPoints = bearingAfter = bearingBefore = modifiers = turns = []
                        previous_routeActive = routeActive
                        print(turnProperties, instructions, turningPoints, bearingAfter, modifiers, turns)

                    #Continue route if active
                    elif routeActive:
                        print("turningPoints")
                        print(turningPoints)

                        #If data is received, continue with navigation
                        if turningPoints:
                            asyncio.create_task(followInstructions(modifiers, turns, bearingAfter, bearingBefore, turningPoints, latitude, longitude))
                            print(f"Current Step: {currentStep}");
                            print(dataToSend)
                            await sendData(dataToSend)

                    previousRouteActive = routeActive;
                    #Inside the while connection is not None loop
                    await asyncio.sleep(0.5)  #Yield control to the event loop
                
                else:
                    print('Route not started/Destination Reached')

            except Exception as e:
                    print(f"An error occurred: {e}")
        else:
            print("No bluetooth connection")

#Peripheral task to advertise BLE service and start main
async def peripheral_task():
    print("Trying to connect")
    while True:
        print("Trying to connect 2")
        try:
            async with await aioble.advertise(
                _ADV_INTERVAL_MS,
                name="ESP32-GPS",
                services=[_BLE_SERVICE_UUID],
            ) as connection:
                if connection:
                    print(f"Connection established with {connection.device}")
                    #Set the on_write handler for the LED characteristic
                    await main(connection)
                    
                    #Wait for the connection to disconnect
                    await connection.disconnected()
                    print("Connection disconnected.")
                else:
                    print("Failed to establish connection")
        except asyncio.CancelledError:
            print("Peripheral task cancelled")
        except Exception as e:
            print("Error in peripheral_task:", e)
        finally:
            await asyncio.sleep_ms(100)  #Allow some time before retrying

async def readSensors():
    global compass_data, gps_data
    while True:
        t3 = asyncio.create_task(readCompass())
        t4 = asyncio.create_task(readGPS())
        await asyncio.gather(t3, t4)
        
        if compass_data is not None and gps_data is not None:
            print("Compass data: ", compass_data)
            print("GPS data: ", gps_data)
        
            combinedData = {
                "type": "currentCoords",
                "azimuth": compass_data.get("azimuth"),
                "direction": compass_data.get("direction"),
                "latitude": gps_data.get("latitude"),
                "longitude": gps_data.get("longitude")
            }
            
            print(combinedData)
        
            await sendData(combinedData)
            
            print("Sent data")
            await asyncio.sleep(1)
        else:
            print("No data yet")
            
#Main task - all happening simultaneously
async def start():
    await prepare()
    #Start advertising and GPS data task
    t1 = asyncio.create_task(peripheral_task())  #BLE peripheral task
    t2 = asyncio.create_task(wait_for_write())  #Getting data from the web server
    t3 = asyncio.create_task(readSensors())
    #Set the `on_write` handler to process incoming writes to the characteristic
    await asyncio.gather(t1,t3)

if __name__ == "__main__":
    try:
        print("MAC Address:", formatted_mac)
        asyncio.run(start())
    except KeyboardInterrupt:
        print("Program terminated.")		
