import urequests
import socket
import time
import math
import machine
import json
from micropyGPS import MicropyGPS
from machine import I2C, Pin
from qmc5883L import QMC5883L
import boot

#Start receiving routeActive status in the background
#import _thread

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

#Send current coords using socket
def sendCurrentCoords(latitude, longitude, azimuth, direction):
    if boot.client_socket is None:
        print("Error: boot.client_socket is not initialized.")
        return  #Exit the function if client_socket is None
    #Format message to indicate this is a coordinates update
    payload = {
        "type": "Coords",	#Short for coordinates
        "latitude": latitude,
        "longitude": longitude,
        "azimuth": azimuth,
        "direction": direction
    }
    print(payload)
    #Convert the dictionary to a JSON string and then encode it to bytes
    message = json.dumps(payload).encode()
    #Send the message over socket
    boot.client_socket.send(message)
    print("Sent:", message)
    #Receive a response from the server for checking
    response = boot.client_socket.recv(1024).decode()
    print("Received from server:", response)

#Use socket to receive routeActive command
def receiveRouteActive():
    global routeActive
    url = f'http://{SERVER_IP}:{port}/get-routeActive'  #Adjust the IP as needed
    try:
        response = urequests.get(url)
        if response.status_code == 200:
            data = response.json()	#Parse the JSON data
            routeActive = data.get("routeActive")
            print(f"routeActive: {routeActive}")
        response.close()
    except Exception as e:
        print(f"Error fetching routeActive: {e}")

def receiveRouteData():
    global startLat, startLon, endLat, endLon
    url = f'http://{SERVER_IP}:{port}/get-routeData'  #Adjust the IP as needed
    print(url)
    response = urequests.get(url)
    try:
        if response.status_code == 200:
            data = response.json()	#Parse the JSON data
            
            #Extract attributes
            startLat = data.get('startLat')
            startLon = data.get('startLon')
            endLat = data.get('endLat')
            endLon = data.get('endLon')
            
            print(f"Start Latitude: {startLat}, Start Longitude: {startLon}")
            print(f"End Latitude: {endLat}, End Longitude: {endLon}")
            
            return startLat, startLon, endLat, endLon  #Return the values if needed
        else:
            print(f"Error getting route data: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while fetching route data: {e}")
    return None, None, None, None  #Return None if there's an error

def receiveNavData():
    global turningPoints, instructions, turnProperties, bearingAfter, modifiers, turns
    url = f'http://{SERVER_IP}:{port}/get-navData'  #Adjust the IP as needed
    print(url)
    response = urequests.get(url)
    try:
        if response.status_code == 200:
            data = response.json()	#Parse the JSON data
            
            turningPoints = data.get('waypoints')
            instructions = data.get('instructions')
            turnProperties = data.get('turnProperties')
            print(f"Turning points: {turningPoints}, Instructions: {instructions}, turnPropertiess: {turnProperties}")
            
            for element in turnProperties:
                bearingAfter.append(element['bearingAfter'])
                modifiers.append(element['modifier'])
                turns.append(element['turns'])
            
            return turningPoints, turnProperties, bearingAfter, modifiers, turns, instructions
        else:
            print(f"Error getting nav data: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while fetching nav data: {e}")
    return None, None, None, None  #Return None if there's an error

#Receive bearing with locations using HTTP
def receive_routeBearing_from_flask():
    try:
        url = f'http://{SERVER_IP}:{port}/get_bearingWLocations'  #Adjust the IP as needed
        response = urequests.get(url)
        if response.status_code == 200:
            bearingWLocations = response.json()
            return bearingWLocations
        else:
            print('Failed to fetch coordinates. Status code:', response.status_code)
            return None
    except Exception as e:
        print('Error fetching coordinates:', e)
        return None

#Send vehicle current step/instruction using HTTP
def sendCurrentStep(action):
    #Format message to indicate this is a coordinates update
    payload = {
        "type": "action",	#Short for coordinates
        "action": action
    }
    #Convert the dictionary to a JSON string and then encode it to bytes
    message = json.dumps(payload).encode()
    #Send the message over socket
    boot.client_socket.send(message)
    print("Sent:", message)
    #Receive a response from the server for checking
    response = boot.client_socket.recv(1024).decode()
    print("Received from server:", response)
    
def sendTestStep():
    global step
    payload = {
        "type": "testStep",	#Short for coordinates
        "testStep": step
    }
    #Convert the dictionary to a JSON string and then encode it to bytes
    message = json.dumps(payload).encode()
    #Send the message over socket
    boot.client_socket.send(message)
    print("Sent:", message)
    step+=1
    #Receive a response from the server for checking
    response = boot.client_socket.recv(1024).decode()
    print("Received from server:", response)

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
        
        #sendCurrentStep(action)
        moveVehicle(msg)	#Function to actually move vehicle according to action
        
    return currentStep, reachedDestination

#Send message to Jetson to move vehicle
def moveVehicle(msg):
    global client_socket_jetson
    
    if client_socket_jetson is None:
        print("Jetson connection not established. Trying to reconnect...")
        boot.connect_to_jetson()
    
    if client_socket_jetson:
        try:
            #Send message
            client_socket_jetson.send(msg.encode())
            print("Sent:", msg)
            
            #Wait for response
            response = client_socket_jetson.recv(1024).decode()
            print("Received from Jetson:", response)
            
            #Delay before the next message
            time.sleep(5)
        except socket.error as e:
            print(f"Communication error: {e}")
            client_socket_jetson.close()
            client_socket_jetson = None
            time.sleep(5)  #Wait before sending another message

#Main loop
def main():
    global latitude, longitude, azimuth, direction, startLat, startLon, endLat, endLon
    global currentStep, reachedDestination, modifiers, bearingAfter, turns, step, matchedBearing, routeActive, previousRouteActive
    i = 0
    step = 0
    matchedBearing = False  #Initialise it to False at the start
    routeActive = False	#Initialise it to False at the start
    previousRouteActive = False
    
    boot.connect_to_wifi()
    boot.connect_to_socket()
    #boot.connect_to_jetson()
    reset_coordinates()
    #calibrateCompass()
    
    #_thread.start_new_thread(receiveRouteActive, ())	#Start background thread to receive routeActive status
    
    while True:
        try:
        #data = gps_serial.read()
            #for byte in data:
                #stat = my_gps.update(chr(byte))
                #if stat is not None:
                    #Compass
                    #azimuth, direction = readCompass()
                    receiveRouteActive()
                    latitude = lat[step]
                    longitude = lon[step]
                    azimuth = 57.06517
                    azimuth = round(azimuth, 4)
                    direction = qmc5883.get_cardinal_direction(azimuth)
                    if step >= len(lat):
                        step = 0  #Reset index to start over
                    #GPS logic
                    #latitude, longitude, satellites, precision = readGPS()
                        
                    print(latitude, longitude, azimuth, direction)
                    sendCurrentCoords(latitude, longitude, azimuth, direction)
                    print('testStep')
                    
                    if not reachedDestination:
                        startLat, startLon, endLat, endLon = receiveRouteData()
                        print('received')
                        
                        time.sleep(2)
                        sendTestStep()
                        
                        #moveVehicle(msg)
                        
                        #Check for route stop
                        if not routeActive and previousRouteActive:  #routeActive was True, now it's False
                            print("Route stopped. Resetting instructions and turning points.")
                            turnProperties = 0
                            instructions = turningPoints = bearingAfter = modifiers = turns = []
                            previous_routeActive = routeActive
                            print(turnProperties, instructions, turningPoints, bearingAfter, modifiers, turns)

                        #Continue route if active
                        elif routeActive:
                            #Receive navigation data if the route is active
                            turningPoints, turnProperties, bearingAfter, modifiers, turns, instructions = receiveNavData()

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
            
#Run the main loop
main()