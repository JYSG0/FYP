import urequests
import socket
import time
import asyncio
import math
import machine
import json
from time import sleep, localtime
from micropyGPS import MicropyGPS
from machine import I2C, Pin
from qmc5883L import QMC5883L
import boot

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

port = '5501'
SERVER_IP = "192.168.229.96"  # Replace with actual server IP

#Variables
instructions = []
turnAngle = []
bearingWLocations = []
modifiers = []
turns = []
bearingAfter = []
step = 0    #Compare turning directions
bearingStep = 0
currentStep = 0
turnStep = 0
cardinalDirections = ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest']

#Boolean checks
roadCurrent = False
roadNext = False
inBoundary = False
route_printed = False
calibrate = False
matchedBearing = False
reachedDestination = False
deviated = False

lat = [40.71271, 40.712802, 40.712823, 40.712856, 40.712928, 40.71304]
lon = [-74.005512, -74.005695, -74.005715, -74.005735, -74.005879, -74.005786]

SERVER_IP = "192.168.167.96"  # Replace with actual server IP
SERVER_PORT = 8765  

#Calibrate compass
def calibrateCompass():
    print("Calibrating... Move the sensor around.")
    x_offset, y_offset, z_offset, x_scale, y_scale, z_scale = qmc5883.calibrate()
    print("Calibration complete.")
    print(f"Offsets: X={x_offset}, Y={y_offset}, Z={z_offset}")
    print(f"Scales: X={x_scale}, Y={y_scale}, Z={z_scale}")
    
#Function to read compass values
def readCompass():
    x , y, z, _ = qmc5883.read_raw()  #Read raw values of compass
    x_calibrated, y_calibrated, _ = qmc5883.apply_calibration(x, y, z)  #Calibrated values from offset and scale
    azimuth = qmc5883.calculate_azimuth(x_calibrated, y_calibrated)
    direction = qmc5883.get_cardinal_direction(azimuth)
    
    return azimuth, direction

#Function to read GPS values
def readGPS():
    latitude_str = my_gps.latitude_string()
    longitude_str = my_gps.longitude_string()

    # Parse latitude
    lat_deg = int(latitude_str.split('°')[0])
    lat_min = float(latitude_str.split('°')[1].split("'")[0])
    lat_dir = latitude_str[-1]  # Get direction (N/S)

    # Parse longitude
    lon_deg = int(longitude_str.split('°')[0])
    lon_min = float(longitude_str.split('°')[1].split("'")[0])
    lon_dir = longitude_str[-1]  # Get direction (E/W)

    # Convert to decimal degrees
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
    global startLat, startLon, endLat, endLon  # Ensure to declare them as global
    startLat = 0.0  # or your desired initial value
    startLon = 0.0  # or your desired initial value
    endLat = 0.0    # or your desired initial value
    endLon = 0.0    # or your desired initial value
    print(f"Coordinates reset to Start: ({startLat}, {startLon}), Destination: ({endLat}, {endLon})")

#Send current coords using socket
def sendCurrentCoords(client_socket, latitude, longitude, azimuth, direction):
    # Format message to indicate this is a coordinates update
    payload = {
        "type": "Coords",	#Short for coordinates
        "latitude": latitude,
        "longitude": longitude,
        "azimuth": azimuth,
        "direction": direction
    }
    # Convert the dictionary to a JSON string and then encode it to bytes
    message = json.dumps(payload).encode()
    #Send the message over socket
    boot.client_socket.send(message)
    print("Sent:", message)
    #Receive a response from the server for checking
    response = boot.client_socket.recv(1024).decode()
    print("Received from server:", response)

def receiveRouteData():
    try:
        response = requests.get('http://{SERVER_IP}:{SERVER_PORT}/get-routeData')
        if response.status_code == 200:
            global startLat, startLon, endLat, endLon
            data = response.json()	#Parse the JSON data
            
            #Extract attributes
            startLat = data.get('startLat')
            startLon = data.get('startLon')
            endLat = data.get('endLat')
            endLon = data.get('endLon')
            
            print(f"Start Latitude: {startLat}, Start Longitude: {startLon}")
            print(f"End Latitude: {endLat}, End Longitude: {endLon}")
            
            return startLat, startLon, endLat, endLon  # Return the values if needed
        else:
            print(f"Error getting route data: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while fetching route data: {e}")
    return None, None, None, None  # Return None if there's an error


def receiveNavData():
    try:
        response = requests.get('http://{SERVER_IP}:{SERVER_PORT}/get-navData')
        if response.status_code == 200:
            global startLat, startLon, endLat, endLon
            data = response.json()	#Parse the JSON data
            
            turningPoints = data.get('waypoints')
            instructions = data.get('instructions')
            modifier = data.get('modifier')
            turns = data.get('turns')
            turnAngle = data.get('turnAngle')
            bearingAfter = None
            
            return turningPoints, turnAngle, bearingAfter, modifier, turns, instructions
        else:
            print(f"Error getting nav data: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while fetching nav data: {e}")
    return None, None, None, None  # Return None if there's an error
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
def sendCurrentStep(step):
    """Send current step to the server."""
    message = f"CurrentStep: {step}"
    boot.client_socket.send(message.encode())
    print("Sent:", message)
    response = boot.client_socket.recv(1024).decode()
    print("Received from server:", response)

#Check if vehicle orientation is right before mobing onto next step
def checkAngleCorrection(azimuth, target_bearing, threshold):
    """Check if the azimuth is within the threshold of the target bearing and determine the correction direction."""
    # Normalize azimuth for comparison
    normalized_azimuth = (azimuth + 360) % 360
    
    # Define lower and upper bounds for the target bearing
    lower_bound = (target_bearing - threshold + 360) % 360
    upper_bound = (target_bearing + threshold + 360) % 360

    if lower_bound <= normalized_azimuth <= upper_bound:
        return True, "Direction matches! Stop turning."
    else:
        # Determine which way to turn to correct
        normalized_difference = (normalized_azimuth - target_bearing + 180) % 360 - 180
        if normalized_difference > threshold:  # Too far right
            return False, "Turn left to correct direction."
        elif normalized_difference < -threshold:  # Too far left
            return False, "Turn right to correct direction."
        else:
            return False, "Keep turning to align."

#Compare starting direction and turn directions to route turn angles
def compareBearing(instructions, directions, azimuth, turnAngle, matchedBearing=False, matchedStartBearing=False, threshold=5.0):
    global step, prev_angle_difference  # Use the global variable for tracking
    startDir = None
    
    if currentStep < 1:
        # Extract starting instruction to face start of route
        moveToStart = instructions[step]
        instructionLower = moveToStart.lower()
        
        for dir in cardinalDirections:
            if dir in instructionLower:
                startDir = dir

        if startDir:
            print(f"Current direction: {directions}, Target direction: {startDir}")
            # Check if azimuth matches the starting direction
            target_bearing = bearingAfter[step]  # Get target bearing for the start direction
            matchedStartBearing, action = checkAngleCorrection(azimuth, target_bearing, threshold)

            if matchedStartBearing:
                step += 1  # Move to the next step
            print(action)
            return matchedBearing, action, matchedStartBearing
        else:
            print("Starting direction not found in the instruction.")
            return

    else:
        # Existing logic for checking ongoing turns
        current_angle_difference = azimuth - turnAngle[step]
        normalized_angle_difference = (current_angle_difference + 180) % 360 - 180
        
        matchedBearing, action = checkAngleCorrection(azimuth, turnAngle[step], threshold)

        # Check previous and current differences
        if abs(normalized_angle_difference) < abs(prev_angle_difference):
            print("Getting closer to the target direction.")
        else:
            print("Getting further away from the target direction.")

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
def followInstructions(modifiers, instructions, turns, direction, bearingAfter, turningPoints, currentLat, currentLon, turnAngle, threshold=5.0):  #Check back to see if need bearingWLocations
    global currentStep, reachedDestination, turnStep
    
    if currentStep >= len(turningPoints):
        print("Destination reached, stop vehicle")
        reachedDestination = True
        msg = "arrive"
        return  reachedDestination
    
    else:
        reachedDestination = False
        
        print(currentStep)
        #Fetch the current turning point based on currentStep
        turnLon, turnLat = turningPoints[currentStep]
        
        #Print current coordinates and the target turning point
        print(f"Current Location: Latitude: {currentLat}, Longitude: {currentLon}")
        print(f"Next Turning Point: Latitude: {turnLat}, Longitude: {turnLon}")
        
        #Calculate the distance to the current turning point
        _, distance_to_turn = haversine_Bearing(currentLat, currentLon, turnLat, turnLon)

        print("")
        print(distance_to_turn)	#debugging line
        
        if distance_to_turn <= threshold:
            matchedBearing, msg = compareBearing(instructions, direction, turnAngle, bearingAfter)
            #We are close enough to the turning point, proceed with the action
            action = f"Action: {instructions[currentStep]} (Time to turn!)"
            
            #If the instruction has the word 'Turn' in it, use turnAngle dictionary
            if 'turn' in instructions[currentStep].lower() or 'turn'in turns[currentStep]:	#Change 'Turn' to lowercase for easier checking
                turn = turnAngle[turnStep]

                #Fetch the elements in the current turnAngle array index
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
            
            sendCurrentStep(currentStep)
            
            currentStep += 1  #Move to the next instruction

        elif currentStep > 0:
            #Not close enough to the turning point yet, keep going straight
            action = f"Keep going straight. Distance to next turn: {distance_to_turn:.2f} meters."
            msg = "Go straight"
            print(action)
        
        moveVehicle(msg)	#Function to actually move vehicle according to action
        
    return currentStep, reachedDestination

#Check if while on route, vehicle deviates
def checkDeviation(currentLat, currentLon, azimuth, bearingWLocations, deviated, deviationStart = None, prevDistance = None):
    global bearingStep
    
    bearings = [item['bearing'] for item in bearingWLocations]	#Extract bearings from bearingWLocations
    location = [item['locations'] for item in bearingWLocations]	#Extract location of bearings from bearingWLocations
    
    if bearingStep >= len(bearings):
        print("Destination reached, stop vehicle")
        
        #Logic to stop vehicle
    else:
        print(bearingStep)
        
        #Thresholds
        bearingThreshold = 5	#Degrees
        distanceThreshold = 5 #Metres
        
        #Fetch the current bearing points based on bearingStep
        targetLon, targetLat = location[bearingStep]
        
        #Print currentlocation and target points
        print(f"Current Location: Latitude: {currentLat}, Longitude: {currentLon}")
        print(f"Bearing Location to note: Latitude: {targetLat}, Longitude: {targetLon}")
        
        #Calculate distance from 
        _, distanceToTarget = haversine_Bearing(currentLat, currentLon, targetLat, targetLon)
        
        #If the distance decreases, it means it hasn't passed. If the distance increases, it means it has passed
        
        if prevDistance is not None and distanceToTarget > prevDistance:
            print(f"Vehicle passed target {bearingStep}. Move to next target")
            bearingStep+=1
            prevDistance = None
        else:
            prevDistance = distanceToTarget
        
        bearingDiff = abs(azimuth - bearings[bearingStep])
        
        if distanceToTarget <= distanceThreshold and bearingDiff <= bearingThreshold:
            deviated = False
            print(f"On course to target {bearingStep}, {targetLat}, {targetLon}")
        
        if distanceToTarget > distanceThreshold and bearingDiff > bearingThreshold:
            deviated = True
            print(f"Deviation detected from target {bearingStep}")
            deviationStart = time.time()
            
    return deviationStart, deviated, bearingStep, prevDistance

#Reorientate vehcle if vehicle 
def reOrientateVehicle(azimuth, bearingWLocations, deviationStart, bearingStep, currentLat, currentLon):
    #Nearest bearing and location
    nearestBearing = bearingWLocations[bearingStep]['bearing']
    nearestLocation = bearingWLocations[bearingStep]['locations']
    targetLon, targetLat = nearestLocation
    
    #Calculate bearing difference
    bearingDiff = abs(azimuth - nearestBearing)
    
    if bearingDiff > 180:
        bearingDiff = 360 - bearingDiff
    
    #Determine left or right
    if azimuth < nearestBearing:
        turnDirection = "right"
    else:
        turnDirection = "left"
        
    #Calculate the time duration for turning (you can adjust the factor here)
    turningDuration = bearingDiff / 2  #Assuming 1 degree takes 1 second to turn
    reOriTime = turningDuration  #Set countdown timer to the calculated duration
    
    startTime = time.time()  #Capture the starting time for reorientation

    #Countdown timer while reOrientation time is greater than 0
    while reOriTime > 0:
        currentTime = time.time()
        elapsedTime = currentTime - startTime
        reOriTime = turningDuration - elapsedTime  #Update remaining time
        
        if reOriTime < 0:
            reOriTime = 0  #Ensure it doesn't go below zero
        
        print(f"Turning {turnDirection} by {bearingDiff} degrees. Time remaining: {reOriTime:.2f} seconds")
        
        #Logic to turn vehicle in the direction (adjust servo motors, etc.)

    #Logic to finish the turn and orient the vehicle to the original bearing
    print("Reorientation complete. Vehicle should be back on course.")

#Send message to Jetson to move vehicle
def moveVehicle(message):
    # Send a message
    boot.client_socket.send(message.encode())
    print("Sent:", message)
        
    # Wait for response
    data = boot.client_socket.recv(1024).decode()
    print("Received from server:", data)
    time.sleep(5)  # Wait before sending another message

#Main loop
def main():
    global latitude, longitude, azimuth, direction, startLat, startLon, endLat, endLon, currentStep, reachedDestination, modifiers, bearingAfter, turns
    i = 0
    reset_coordinates()
    
    global matchedBearing  #Declare matchedBearing as global
    matchedBearing = False  #Initialize it to False at the start
    
    while not reachedDestination:
        try:
            while True:
                #data = gps_serial.read()
                #for byte in data:
                    #stat = my_gps.update(chr(byte))
                    #if stat is not None:
                        #Compass
                        #azimuth, direction = readCompass()
                        latitude = lat[i]
                        longitude = lon[i]
                        azimuth = 0
                        direction = qmc5883.get_cardinal_direction(azimuth)
                        i += 1
                        if i >= len(lat):
                            i = 0  # Reset index to start over
                        #GPS logic
                        #latitude, longitude, satellites, precision = readGPS()
                    
                        sendCurrentCoords(boot.client_socket, latitude, longitude, azimuth, direction)
                        startLat, startLon, endLat, endLon = receiveRouteData()

                        if endLat and endLon:
                            turningPoints, turnAngle, bearingAfter, modifiers, turns, instructions = receiveNavData()
                            bearingWLocations = receive_routeBearing_from_flask()	#Should contain both routeBearings and routeBearingsLocations
                            
                            if turningPoints and instructions and turnAngle:
                                currentStep = 0
                                
                                matchedBearing, action, matchedStartBearing = compareBearing(instructions, direction, azimuth, turnAngle, bearingAfter)
                                
                                print(f"2. {matchedBearing}, {action}, {matchedStartBearing}")

                                if matchedStartBearing:
                                    currentStep, reachedDestination = followInstructions(modifiers, instructions, turns, direction, bearingWLocations, turningPoints, latitude, longitude, turnAngle)
                                else:
                                    print(f"No instructions to follow yet")
                                
                                #deviationStart, deviated, bearingStep, prevDistance = checkDeviation(latitude, longitude, azimuth, bearingWLocations, deviated)
                                
                                #if deviated:
                                 #   reOrientateVehicle(azimuth, bearingWLocations, deviationStart, bearingStep, latitude, longitude)
                        
                        time.sleep(0.5)	#delay to not overwhelm esp32
    
        except Exception as e:
            print(f"An error occurred: {e}")
            
#Run the main loop
main()
