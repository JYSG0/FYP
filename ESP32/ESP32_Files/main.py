import urequests
import socket
import time
import math
import machine
import json
import asyncio
import aioble
import bluetooth
from micropyGPS import MicropyGPS
import ujson
from micropython import const
from machine import I2C, Pin
from qmc5883L import QMC5883L
import struct

_BLE_SERVICE_UUID = bluetooth.UUID('19b10000-e8f2-537e-4f6c-d104768a1214')
_BLE_SENSOR_CHAR_UUID = bluetooth.UUID('19b10001-e8f2-537e-4f6c-d104768a1214')  # GPS Data UUID
_BLE_LED_UUID = bluetooth.UUID('19b10002-e8f2-537e-4f6c-d104768a1214')

_ADV_INTERVAL_MS = 250_000

# Register GATT service and characteristics
ble_service = aioble.Service(_BLE_SERVICE_UUID)
sensor_characteristic = aioble.Characteristic(ble_service, _BLE_SENSOR_CHAR_UUID, read=True, notify=True)
led_characteristic = aioble.Characteristic(ble_service, _BLE_LED_UUID, read=True, write=True, notify=True, capture=True)

# Register services
aioble.register_services(ble_service)

#Instantiate the micropyGPS object
my_gps = MicropyGPS()

#Define the UART pins and create a UART object
#gps_serial = machine.UART(2, baudrate=9600, tx=17, rx=16)	#gps rx ->SD2 = GPIO 9,  gps tx ->SD3 = GPIO 10

#Define the SCL and SDA pins
i2c = I2C(1, scl=Pin(22), sda=Pin(21), freq=100000)
#Place pins into qmc5883 compass library file
qmc5883 = QMC5883L(i2c)

#Set up HERE API - 250,000 requests per month
api_key = 'SQhcQxSqNhmFGy1cf3vFZP6sUx69OuhkFrWiuHigA-E'

port = '5501'	#HTTP port for webserver
SERVER_IP = "192.168.230.96"  #Replace with actual server IP

#Variables
instructions = []
turnProperties = []
bearingWLocations = []
modifiers = []
turns = []
bearingAfter = []
step = 0    #Compare turning directions
currentStep = 0
turnStep = 0
cardinalDirections = ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest']
prev_angle_difference = float('inf')  #Initial large difference

#Boolean checks
matchedBearing = False
matchedStartBearing = False
reachedDestination = False
routeActive = False
previousRouteActive = False

lon = [-74.004288, -74.004917, -74.006252, -74.006204, -74.006415, -74.00735]
lat = [40.713123, 40.713438, 40.714023, 40.714127, 40.714225, 40.713103]
latitude = 40.713123
longitude = -74.004288
azimuth = 20
direction = 'North'
message = ['test', 'test1', 'start', 'forward', 'backward', 'stop','start', 'forward', 'backward', 'stop']

#Calibrate compass
def calibrateCompass():
    print("Calibrating... Move the sensor around.")
    x_offset, y_offset, z_offset, x_scale, y_scale, z_scale = qmc5883.calibrate()
    print("Calibration complete.")
    print(f"Offsets: X={x_offset}, Y={y_offset}, Z={z_offset}")
    print(f"Scales: X={x_scale}, Y={y_scale}, Z={z_scale}")
    
#Function to read compass values
def readCompass():
    global azimuth, direction
    x , y, z, _ = qmc5883.read_raw()  #Read raw values of compass
    x_calibrated, y_calibrated, _ = qmc5883.apply_calibration(x, y, z)  #Calibrated values from offset and scale
    azimuth = qmc5883.calculate_azimuth(x_calibrated, y_calibrated)
    direction = qmc5883.get_cardinal_direction(azimuth)
    
    return azimuth, direction

#Function to read GPS values
def readGPS():
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
    satellites = my_gps.satellites_in_use()
    precision = my_gps.hdop()
    
    return latitude, longitude, satellites, precision

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

#Check if vehicle orientation is right before mobing onto next step
def checkAngleCorrection(azimuth, target_bearing, threshold):
    """Check if the azimuth is within the threshold of the target bearing and determine the correction direction."""
    #Normalize azimuth for comparison
    normalized_azimuth = (azimuth + 360) % 360
    
    #Define lower and upper bounds for the target bearing
    lower_bound = (target_bearing - threshold + 360) % 360
    upper_bound = (target_bearing + threshold + 360) % 360
    
    print(lower_bound, normalized_azimuth, upper_bound)
    
    if lower_bound <= normalized_azimuth <= upper_bound:
        return True, "Direction matches! Stop turning."
    else:
        #Determine which way to turn to correct
        normalized_difference = (normalized_azimuth - target_bearing + 180) % 360 - 180
        if normalized_difference > threshold:  #Too far right
            return False, "Turn left to correct direction."
        elif normalized_difference < -threshold:  #Too far left
            return False, "Turn right to correct direction."
        else:
            return False, "Keep turning to align."

#Compare starting direction and turn directions to route turn angles
def compareBearing(instructions, directions, azimuth, turnProperties, matchedBearing=False, threshold=5.0):
    global step, prev_angle_difference, matchedStartBearing  #Use the global variable for tracking
    startDir = None
    
    print('currentStep: ',currentStep)
    print('step: ', step)
    
    if step < 1:
        #Extract starting instruction to face start of route
        moveToStart = instructions[step]
        instructionLower = moveToStart.lower()
        
        for dir in cardinalDirections:
            if dir in instructionLower:
                startDir = dir
                #print('')
                #print('startDir: ', startDir)
                #print('')

        if startDir:
            print(f"Current direction: {directions}, Target direction: {startDir}")
            #Check if azimuth matches the starting direction
            target_bearing = bearingAfter[step]  #Get target bearing for the start direction
            matchedStartBearing, action = checkAngleCorrection(azimuth, target_bearing, threshold)

            if matchedStartBearing:
                print(f'matchedStartBearing: {matchedStartBearing}')
                step += 1  #Move to the next step
            print('action: ', action)
            print('step: ',step)
            return matchedBearing, action, matchedStartBearing
        else:
            print("Starting direction not found in the instruction.")
            return

    else:
        print('turnProperties[step]: ',turnProperties[step]['bearingAfter'])
        #Existing logic for checking ongoing turns
        current_angle_difference = azimuth - turnProperties[step]['bearingAfter']
        print('current_angle_difference: ', current_angle_difference)
        normalized_angle_difference = (current_angle_difference + 180) % 360 - 180
        print('normalized_angle_difference:', normalized_angle_difference)
        
        matchedBearing, action = checkAngleCorrection(azimuth, turnProperties[step]['bearingAfter'], threshold)
        print('action2:', action)
        print('matchedBearing', matchedBearing)
        print('')
        #Check previous and current differences
        if not matchedBearing:
            if abs(normalized_angle_difference) < abs(prev_angle_difference):
                print("Getting closer to the target direction.")
            else:
                print("Getting further away from the target direction.")
        print('')
        #Update prev_angle_difference for the next iteration
        prev_angle_difference = normalized_angle_difference
        return matchedBearing, action, matchedStartBearing

#Calculate bearing and distance between 2 points on Earth
def haversine_Bearing(lat1, lon1, lat2, lon2): 
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    initial_bearing = math.atan2(x, y)
    compass_bearing = (math.degrees(initial_bearing) + 360) % 360
    
    R = 6371  #Radius of Earth in kilometers
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(lat1) * math.cos(lat2) * 
         math.sin(dlon / 2) ** 2)
    distance = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)) * 1000  #Distance in meters

    return compass_bearing, distance

#Instructions sequence for vehicle on route
def followInstructions(modifiers, instructions, turns, direction, bearingAfter, turningPoints, currentLat, currentLon, turnProperties, threshold=5.0):  #Check back to see if need bearingWLocations
    global currentStep, reachedDestination, turnStep
    
    if currentStep >= len(turningPoints):
        print("Destination reached, stop vehicle")
        reachedDestination = True
        msg = "arrive"
        return  reachedDestination
    
    else:
        reachedDestination = False
        
        print('currentStep: ', currentStep)
        #Fetch the current turning point based on currentStep
        print('turningPoints[currentStep]: ', turningPoints[currentStep])
        turnLon, turnLat = turningPoints[currentStep]
        print(f'turnLon: {turnLon}, turnLat: {turnLat}')
        
        #Print current coordinates and the target turning point
        print(f"Current Location: Latitude: {currentLat}, Longitude: {currentLon}")
        print(f"Next Turning Point: Latitude: {turnLat}, Longitude: {turnLon}")
        
        #Calculate the distance to the current turning point
        _, distance_to_turn = haversine_Bearing(currentLat, currentLon, turnLat, turnLon)

        print("")
        print(distance_to_turn)	#debugging line
        
        if distance_to_turn <= threshold:
            #We are close enough to the turning point, proceed with the action
            #print(f"Action: {instructions[currentStep]} (Time to turn!)")
            action = f"Action: {instructions[currentStep]} (Time to turn!)"
            #matchedBearing, msg, _  = compareBearing(instructions, direction, turnProperties, bearingAfter)
            #matchedBearing, action, matchedStartBearing
            #print('msg: ', msg)
            
            #If the instruction has the word 'Turn' in it, use turnProperties dictionary
            if 'turn' in instructions[currentStep].lower() or 'turn'in turns[currentStep]:	#Change 'Turn' to lowercase for easier checking
                turn = turnProperties[turnStep]

                #Fetch the elements in the current turnProperties array index
                angle = turn['angle']
                #location = turn['location']
                turnOri = turn['turnOri']

                msg = f"turn {turnOri} {angle}"

                if not matchedBearing:
                    msg = f"{msg}, keep turning"
                else:
                    turnStep+=1	#Move to get ready for the next turn angle
                
                #action = f"{action} for {angle} degrees, {turnOri}"
            
            elif 'depart' in modifiers[currentStep] and matchedBearing:
                msg = "depart"

            elif 'fork' in modifiers[currentStep]:
                msg = f"fork {turnOri}"

            print(action)
            
            currentStep += 1  #Move to the next instruction
            
            print(currentStep)

        elif currentStep > 0:
            #Not close enough to the turning point yet, keep going straight
            action = f"Keep going straight. Distance to next turn: {distance_to_turn:.2f} meters."
            msg = "Go straight"
            print(action)
        
        await sensor_characteristic.write(action)
        moveVehicle(msg)	#Function to actually move vehicle according to action
        
    return currentStep, reachedDestination

#Send message to Jetson to move vehicle
def moveVehicle(msg):
    data = {
        'type': 'msg',  # The action you want the vehicle to perform
        'msg': msg  # Any other data you want to send
    }
    await sensor_characteristic.write(data)
            
#Helper to encode GPS data, including azimuth and direction
def formatGPSdata(latitude, longitude, azimuth, direction):
    gps_data = {
        "latitude": latitude,
        "longitude": longitude,
        "azimuth": azimuth,
        "direction": direction
    }
    return ujson.dumps(gps_data).encode('utf-8')

#Main loop
def prepare():
    global latitude, longitude, azimuth, direction, startLat, startLon, endLat, endLon, lat, lon
    global currentStep, reachedDestination, modifiers, bearingAfter, turns, step, matchedBearing, routeActive, previousRouteActive
    i = 0
    step = 0
    matchedBearing = False  #Initialise it to False at the start
    routeActive = False	#Initialise it to False at the start
    previousRouteActive = False
    
    # Run the main task
    #calibrateCompass()

data_buffer = bytearray()

# Function to decode and process the received JSON data
def _decode_data(data):
    try:
        print(f"Decoding JSON data: {data}")
        # Process the JSON data (e.g., control LED based on data)
        json_data = ujson.loads(data)
        print(f"Decoded JSON: {json_data}")
        formatJSONbleak(json_data)
    except Exception as e:
        print(f"Error decoding received data: {e}")

# Task to handle incoming writes (handling fragmentation)
async def wait_for_write():
    while True:
        try:
            # Wait for data to be written to the characteristic
            connection, data = await led_characteristic.written()
            
            if data:
                is_last_chunk, chunk = struct.unpack("B", data[0:1])[0], data[1:]
                data_buffer.extend(chunk)  # Add the chunk to the buffer

                print(f"Received chunk, is_last_chunk={is_last_chunk}")
                
                # Check if it's the last chunk and if we've received the full message
                if is_last_chunk:
                    # Reassemble the full message
                    full_data = data_buffer.decode('utf-8')
                    print(f"Full data received: {full_data}")
                    
                    # Reset the buffer for future data
                    data_buffer.clear()

                    # Process the received JSON data
                    _decode_data(full_data)
                else:
                    print("Waiting for more chunks...")
        except Exception as e:
            print(f"Error receiving data: {e}")

# Helper to encode GPS data, including azimuth and direction
def formatGPSdata(latitude, longitude, azimuth, direction):
    gps_data = {
        "latitude": latitude,
        "longitude": longitude,
        "azimuth": azimuth,
        "direction": direction
    }
    return ujson.dumps(gps_data).encode('utf-8')  # Encode JSON as bytes

# Task to send GPS data periodically
async def main(connection):
    global latitude, longitude, azimuth, direction, startLat, startLon, endLat, endLon
    global currentStep, reachedDestination, modifiers, bearingAfter, turns, step, matchedBearing, routeActive, previousRouteActive
    routeActive = False
    previousRouteActive = True
    currentStep = 0

    try:
        #data = gps_serial.read()
        #for byte in data:
            #stat = my_gps.update(chr(byte))
            #if stat is not None:
                #azimuth, direction = readCompass()
                latitude = lat[step]
                longitude = lon[step]
                msg = message[step]
                azimuth = 57.06517
                azimuth = round(azimuth, 4)
                direction = qmc5883.get_cardinal_direction(azimuth)
                if step >= len(lat):
                    step = 0  #Reset index to start over
                #GPS logic
                #latitude, longitude, satellites, precision = readGPS()
                    
                print(latitude, longitude, azimuth, direction)
                gps_data = formatGPSdata(latitude, longitude, azimuth, direction)
                if sensor_characteristic is not None:
                # Send the encoded GPS data
                    try:
                        print(f"Sending GPS data: {latitude}, {longitude}, {azimuth}, {direction}")  # Debugging log
                        print(gps_data)
                        await sensor_characteristic.write(gps_data)
                        #await sensor_characteristic.write(currentStep)
                        
                    except Exception as e:
                        print(f"GPS data: {gps_data}")
                        print(f"Error sending GPS data: {e}")
                else:
                    print("Error: sensor_characteristic is None.")
                    print('testStep')
                
                if not reachedDestination:
                    print('received')
                    
                    time.sleep(2)
                    #await sensor_characteristic.write(currentStep)
                    
                    #moveVehicle(msg)
                    
                    #Check for route stop
                    if not routeActive and previousRouteActive:  #routeActive was True, now it's False
                        print("Route stopped. Resetting instructions and turning points.")
                        turnProperties = 0
                        #instructions = turningPoints = bearingAfter = modifiers = turns = []
                        #previous_routeActive = routeActive
                        #print(turnProperties, instructions, turningPoints, bearingAfter, modifiers, turns)

                    #Continue route if active
                    elif routeActive:
                        #Receive navigation data if the route is active
                        print("turningPOints")
                        #turningPoints, turnProperties, bearingAfter, modifiers, turns, instructions = receiveNavData()

                        #If data is received, continue with navigation
                        if turningPoints and instructions and turnProperties:
                            matchedBearing, action, matchedStartBearing = compareBearing(
                                instructions, direction, azimuth, turnProperties, bearingAfter
                            )
                            
                            if matchedStartBearing:
                                currentStep, reachedDestination = followInstructions(
                                    modifiers, instructions, turns, direction, bearingWLocations, 
                                    turningPoints, latitude, longitude, turnProperties
                                )
                                print(f"Current Step: {currentStep}");
                    previousRouteActive = routeActive;
                    time.sleep(0.5)	#delay to not overwhelm esp32

    except Exception as e:
            print(f"An error occurred: {e}")

# Peripheral task to advertise BLE service
async def peripheral_task():
    while True:
        try:
            async with await aioble.advertise(
                _ADV_INTERVAL_MS,
                name="ESP32-GPS",
                services=[_BLE_SERVICE_UUID],
            ) as connection:
                if connection:
                    print(f"Connection established with {connection.device}")
                    # Set the on_write handler for the LED characteristic
                    while connection is not None:
                        if connection.is_connected():
                            await main(connection)
                    
                    # Wait for the connection to disconnect
                    await connection.disconnected()
                    print("Connection disconnected.")
                else:
                    print("Failed to establish connection")
        except asyncio.CancelledError:
            print("Peripheral task cancelled")
        except Exception as e:
            print("Error in peripheral_task:", e)
        finally:
            await asyncio.sleep_ms(100)  # Allow some time before retrying

# Main task
async def bootTask():
    # Start advertising and GPS data task
    t1 = asyncio.create_task(peripheral_task())  # BLE peripheral task
    t3 = asyncio.create_task(wait_for_write())
    # Set the `on_write` handler to process incoming writes to the characteristic
    await asyncio.gather(t1)

#Initialise variables and call functions that run once
prepare
# Run the event loop
asyncio.run(bootTask())
