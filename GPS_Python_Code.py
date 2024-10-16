#ESP WROOM 32 38 pins

import network
import urequests
import requests
import time
import math
import machine
import json
import machine
from time import sleep, localtime
from micropyGPS import MicropyGPS
from machine import I2C, Pin
from qmc5883L import QMC5883L

#Instantiate the micropyGPS object
my_gps = MicropyGPS()

#Define the UART pins and create a UART object
gps_serial = machine.UART(2, baudrate=9600, tx=9, rx=10)	#SD2 = GPIO 9, SD3 = GPIO 10

#Define the SCL and SDA pins
i2c = I2C(1, scl=Pin(26), sda=Pin(25), freq=100000)
#Place pins into qmc5883 compass library file
qmc5883 = QMC5883L(i2c)

#Set up HERE API - 250,000 requests per month
api_key = 'SQhcQxSqNhmFGy1cf3vFZP6sUx69OuhkFrWiuHigA-E'

#Variables
instructions = []
turnAngle = []
bearingWLocations = []
distance = 0
bearingStep = 0
currentStep = 0
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

latitude = 1.310217
longitude = 103.7776

#Calibrate compass
print("Calibrating... Move the sensor around.")
x_offset, y_offset, z_offset, x_scale, y_scale, z_scale = qmc5883.calibrate()
print("Calibration complete.")
print(f"Offsets: X={x_offset}, Y={y_offset}, Z={z_offset}")
print(f"Scales: X={x_scale}, Y={y_scale}, Z={z_scale}")

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

#Connect to Wi-Fi
def connect_wifi():
    ssid = 'AndroidAP3a93'
    password = 'buhh1927'
    
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(ssid, password)
    
    while not wlan.isconnected():
        print('Connecting to WiFi...')
        time.sleep(1)
    
    print('Connected:', wlan.ifconfig())

def send_current_coordinates_to_flask(latitude, longitude, azimuth, direction):
    url = 'http://192.168.96.96:5500/receive_coordinates'  #Adjust the IP and Flask route as needed
    try:
        #Create payload with the current coordinates
        payload = {
            'current_latitude': latitude,
            'current_longitude': longitude,
            'azimuth': azimuth,
            'direction': direction
        }
        #print("Payload: ", payload)	#debugging line
        headers = {'Content-Type': 'application/json'}
        
        #Send POST request with current coordinates
        response = urequests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            print("Successfully sent current coordinates to Flask.")
            return True  #Indicate success
        else:
            print(f"Failed to send current coordinates. Status code: {response.status_code}")
            return False  #Indicate failure
    except Exception as e:
        print('Error sending current coordinates to Flask:', e)
        return False

def receive_coordinates_from_flask():
    try:
        url = 'http://192.168.96.96:5500/get_coordinates'  #Adjust the IP as needed
        response = urequests.get(url)
        if response.status_code == 200:
            data = response.json()
            startLat = data['start_latitude']
            startLon = data['start_longitude']
            desLat = data['end_latitude']
            desLon = data['end_longitude']
            print(f"Fetched Coordinates - Start: ({startLat}, {startLon}), Destination: ({desLat}, {desLon})")
            return startLat, startLon, desLat, desLon
        else:
            print('Failed to fetch coordinates. Status code:', response.status_code)
            return None
    except Exception as e:
        print('Error fetching coordinates:', e)
        return None

def receive_turningPoints_from_flask():
    try:
        url = 'http://192.168.96.96:5500/get_waypoints'  #Adjust the IP as needed
        response = urequests.get(url)
        if response.status_code == 200:
            turningPoints = response.json()  #Directly assign the response
            return turningPoints  #Return the list of waypoints
        else:
            print('Failed to fetch coordinates. Status code:', response.status_code)
            return None
    except Exception as e:
        print('Error fetching coordinates:', e)
        return None
    
def receive_instructions_from_flask():
    try:
        url = 'http://192.168.96.96:5500/get_instructions'  #Adjust the IP as needed
        response = urequests.get(url)
        if response.status_code == 200:
            instructions = response.json()  #Directly assign the response
            return instructions
        else:
            print('Failed to fetch coordinates. Status code:', response.status_code)
            return None
    except Exception as e:
        print('Error fetching coordinates:', e)
        return None
    
def receive_turnAngle_from_flask():
    try:
        url = 'http://192.168.96.96:5500/get_turnAngle'  #Adjust the IP as needed
        response = urequests.get(url)
        if response.status_code == 200:
            turnAngle = response.json()  #Directly assign the response
            return turnAngle
        else:
            print('Failed to fetch coordinates. Status code:', response.status_code)
            return None
    except Exception as e:
        print('Error fetching coordinates:', e)
        return None

def receive_routeBearing_from_flask():
    try:
        url = 'http://192.168.96.96:5500/get_bearingWLocations'  #Adjust the IP as needed
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

def compareStartBearing(instructions, directions):
    startDir = None
    
    if currentStep < 1:
        #Extract starting instruction to face start of route
        moveToStart = instructions[0]
        #print(moveToStart)	#debugging line
        #Convert string to lowercase for easier matching
        instructionLower = moveToStart.lower()
        
        for dir in cardinalDirections:
            if dir in instructionLower:
                startDir = dir	#Assign the start direction
                
        #print(startDir)	#debugging line
        
        if startDir:
            print(f"Starting direction extracted: {startDir}")
        else:
            print("Starting direction not found in the instruction.")
            return
    
        return startDir  
    
#Calucalate bearing and distance between 2 points on Earth
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

def followInstructions(instructions, direction, turningPoints, currentLat, currentLon, turnAngle, threshold=5.0):
    global currentStep, reachedDestination, turnStep
    
    if currentStep >= len(turningPoints):
        print("Destination reached, stop vehicle")
        reachedDestination = True
        
        #Logic to stop vehicle
        #moveVehicle(action, turnOri, angle, reachedDestination)	#Use reachedDestinaiton to stop the vehcile in moveVehicle function
        
        return  #Exit the function once the destination is reached
    
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
        
        #print("")
        #print(distance_to_turn)	#debugging line
        
        if distance_to_turn <= threshold:
            #We are close enough to the turning point, proceed with the action
            action = f"Action: {instructions[currentStep]} (Time to turn!)"
            
            #If the instruction has the word 'Turn' in it, use turnAngle dictionary
            if 'turn' in instructions[currentStep].lower():	#Change 'Turn' to lowercase for easier checking
                turn = turnAngle[turnStep]
                
                #Fetch the elements in the current turnAngle array index
                angle = turn['angle']
                location = turn['location']
                turnOri = turn['turnOri']
                
                action = f"{action} for {angle} degrees, {turnOri}"
                
                turnStep+=1	#Move to get ready fot the next turn angle
                
            print(action)
            currentStep += 1  #Move to the next instruction
        else:
            #Not close enough to the turning point yet, keep going straight
            action = f"Keep going straight. Distance to next turn: {distance_to_turn:.2f} meters."
            print(action)
            
        #moveVehicle(action, turnOri, angle, reachedDestination)	#Function to actually move vehicle according to action
        
    return currentStep, reachedDestination

def checkDeviation(currentLat, currentLon, azmiuth, bearingWLocations, deviated, deviationStart = None, prevDistance = None):
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
    
#Main loop
def main():
    connect_wifi()
    
    global matchedBearing  #Declare matchedBearing as global
    matchedBearing = False  #Initialize it to False at the start
    
    while True:
        try:
            while gps_serial.any():
                data = gps_serial.read()
                for byte in data:
                    stat = my_gps.update(chr(byte))
                    if stat is not None:
                        #Compass
                        x, y, z, _ = qmc5883.read_raw()  #Read raw values of compass
                        x_calibrated, y_calibrated, z_calibrated = qmc5883.apply_calibration(x, y, z)  #Calibrated values from offset and scale
                        azimuth = qmc5883.calculate_azimuth(x_calibrated, y_calibrated)
                        direction = qmc5883.get_cardinal_direction(azimuth)
                        
                        #GPS logic
                        utc_time = (int(my_gps.timestamp[0]), int(my_gps.timestamp[1]), int(my_gps.timestamp[2]))  # hour, minute, second
                        sgt_time = convert_to_sgt(utc_time)

                        # Get latitude and longitude
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
                        
                        # Print parsed GPS data
                        print('UTC Timestamp:', my_gps.timestamp)
                        print('Date:', my_gps.date_string('long'))
                        print('Latitude (DD):', latitude)
                        print('Longitude (DD):', longitude)
                        print('Horizontal Dilution of Precision:', my_gps.hdop)
                        print()
                        
                        send_current_coordinates_to_flask(latitude, longitude, azimuth, direction)
                        startLat, startLon, desLatitude, desLongitude = receive_coordinates_from_flask()

                        if desLatitude and desLongitude:
                            turningPoints = receive_turningPoints_from_flask()
                            turnAngle = receive_turnAngle_from_flask()
                            instructions = receive_instructions_from_flask()
                            bearingWLocations = receive_routeBearing_from_flask()	#Should contain both routeBearings and routeBearingsLocations
                            
                            if turningPoints and instructions and turnAngle:
                                currentStep = 0
                                bearingStep = 0
                                
                                startDir = compareStartBearing(instructions, direction)
                                
                                print(f"Current direction: {direction}, Target direction: {startDir}")
                                if startDir == direction:
                                    print("Direction matches! Stop turning.")
                                    matchedBearing = True  #Update matchedBearing

                                elif(currentStep < 1):
                                    print("Keep turning! Direction not matched yet")
                                    
                                if matchedBearing:
                                    print(f"matchedBearing: ", matchedBearing)
                                    print(f"turning points: ", turningPoints)
                                    currentStep, reachedDestination = followInstructions(instructions, direction, turningPoints, latitude, longitude, turnAngle)
                                else:
                                    print(f"No instructions to follow yet")
                                
                                deviationStart, deviated, bearingStep, prevDistance = checkDeviation(latitude, longitude, azmiuth, bearingWLocations, deviated)
                                
                                if deviated:
                                    reOrientateVehicle(azimuth, bearingWLocations, deviationStart, bearingStep, latitude, longitude)
                        
                        time.sleep(0.5)	#delay to not overwhelm esp32
    
        except Exception as e:
            print(f"An error occurred: {e}")
            
#Run the main loop
main()
