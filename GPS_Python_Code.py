import machine
import math
import requests
import network
import time
import json
from time import localtime
from micropyGPS import MicropyGPS

# Set up API
api_key = 'SQhcQxSqNhmFGy1cf3vFZP6sUx69OuhkFrWiuHigA-E'

# Instantiate the micropyGPS object
my_gps = MicropyGPS()

# Define the UART pins and create a UART o12bject
gps_serial = machine.UART(2, baudrate=9600, tx=26, rx=25)

#Temporary fixed desLatitude & desLongitude
#desLatitude = 1.31151
#desLongitude = 103.77812

turning_points = []
compassDirection = ["North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest"]

roadCurrent = False
roadNext = False
inBoundary = False

#Set up WiFi connection with esp32
def connect():
    ssid = 'AndroidAP3a93'
    password = 'buhh1927'

    wifi = network.WLAN(network.STA_IF)
    wifi.active(True)
    if not wifi.isconnected():
        print("Connecting to the network...")
        wifi.connect(ssid, password)
        while not wifi.isconnected():
            pass
    print('network config: ', wifi.ifconfig())

#Menu to insert desired destination
def menu():
    global desLatitude, desLongitude
    
    validChoices = ["Address", "Coordinates", "Admin"]
    choice = None
    
    while choice not in validChoices:
        choice = input("Address or Coordinates?: ")
        #print(f"{choice}")
        
        if choice == "Address":
            addressName = input("Enter address: ")
            addressName = addressName.replace(" ", "+")
            return choice, addressName, None, None
        elif choice == "Coordinates":
            try:
                desLatitude = float(input("Please input destination latitude: "))
                desLongitude = float(input("Please input destination longitude: "))
                print(f"Destination coordinates: {desLatitude}, {desLongitude}")
                return choice, None, desLatitude, desLongitude
            
            except ValueError:
                print("Invalid input for latitude or longitude")
                return None, None, None, None
            
        elif choice == "Admin":
            try:
                desLatitude = 1.31158
                desLongitude = 103.77918
                return choice, None, desLatitude, desLongitude
            
            except ValueError:
                print("Invalid input for latitude or longitude")
                return None, None, None, None
            
        else:
            print("Invalid choice")

#Query HERE API to reverse geocode current location to address
def revGeoCode(latitude, longitude):
    try:
        responseUrl = f'https://revgeocode.search.hereapi.com/v1/revgeocode?at={latitude}%2C{longitude}&lang=en-US&apiKey={api_key}'
        response = requests.get(responseUrl)
        if response.status_code == 200:
            currentData = response.json()
            
            #Check if there are 'items'/attributes in response
            if 'items' in currentData and len(currentData['items']) > 0:
                currentAddress = currentData['items'][0]['address']['label']
                return currentAddress
            else:
                print("No items")
                
        else:
            print(f"Error getting HERE API rgc: {response.status_code}")
    except Exception as e:
        print(f"Error occured while getting HERE API rgc: {e}")
    return None

#Address format entered may be different for everybody so using search then taking the correct format to find correct destination format 
def searchAddress(latitude, longitude, addressName):
    global desLatitude, desLongitude
    try:
        searchUrl = f'https://autosuggest.search.hereapi.com/v1/autosuggest?at={latitude}%2C{longitude}&lang=en-US&q={addressName}&country=SG&apiKey={api_key}'
        #print(f"{searchUrl}")
        searchResponse = requests.get(searchUrl)
        if searchResponse.status_code == 200:
            searchData = searchResponse.json()
            
            #Check if there are 'items'/attributes in response
            if 'items' in searchData and len(searchData['items']) > 0:
                searchAddress = searchData['items'][0]['title']
                #print(f"Is {searchAddress} your destination?")
                choiceAddress = input(f"Is {searchAddress} your destination? ")
                
                if choiceAddress == "Yes":
                    if 'items' in searchData and len(searchData['items']) > 0:
                        desLatitude = searchData['items'][0]['access'][0]['lat']
                        print(f"{desLatitude}")
                        desLongitude = searchData['items'][0]['access'][0]['lng']
                        print(f"{desLongitude}")
                        address = searchData['items'][0]['address']['label']
                                
                        return address
                    else:
                        print("No geocode items")
                else:
                    print("Address not confirmed")
            else:
                print("No items")
        else:
            print(f"Error getting HERE API sa: {searchResponse.status_code}")
    except Exception as e:
        print(f"Error occured while getting HERE API sa: {e}")
    return None

#Query HERE API to reverse geocode destination coordinates to readable address
def desRevGeoCode(desLatitude, desLongitude):
    try:
        desResponseUrl = f'https://revgeocode.search.hereapi.com/v1/revgeocode?at={desLatitude}%2C{desLongitude}&lang=en-US&apiKey={api_key}'
        desResponse = requests.get(desResponseUrl)
        if desResponse.status_code == 200:
            destinationData = desResponse.json()
            
            #Check if there are 'items'/attributes in response
            if 'items' in destinationData and len(destinationData['items']) > 0:
                desAddress = destinationData['items'][0]['address']['label']
                return desAddress
            else:
                print("No items")

        else:
            print(f"Error getting HERE API drgc: {desResponse.status_code}")
    except Exception as e:
        print(f"Error occured while getting HERE API drgc: {e}")
    return None

def route(latitude, longitude, desLatitude, desLongitude):
    routeUrl = f"https://router.hereapi.com/v8/routes?transportMode=car&origin={latitude}%2C{longitude}&destination={desLatitude}%2C{desLongitude}&return=polyline%2Cturnbyturnactions&apiKey={api_key}"
    try:
        routeResponse = requests.get(routeUrl)
        if routeResponse.status_code == 200:
            routeData = routeResponse.json()
            #print("Route Data:", routeData)  # Debugging line
            
            if 'routes' in routeData and len(routeData['routes']) > 0:
                # Return the full routeData for further processing
                return routeData
            else:
                print("No routes found in response.")
                return None
        else:
            print(f"Error getting route: {routeResponse.status_code}")
            print("Error Response:", routeResponse.text)  # Print error details
            return None
    except Exception as e:
        print(f"Error occurred while getting route: {e}")
        return None

def decode_polyline(encoded):
    # Constants from the polyline encoding
    ENCODING_TABLE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    DECODING_TABLE = [
        62, -1, -1, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1, -1,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, -1, -1, -1, -1, 63, -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
    ]

    def decode_char(char):
        char_code = ord(char)
        return DECODING_TABLE[char_code - 45]

    def to_signed(val):
        res = val
        if res & 1:
            res = ~res
        return res >> 1

    # Decoding the polyline
    result = []
    index = 0
    length = len(encoded)
    lat = 0
    lng = 0

    while index < length:
        b, shift = 0, 0
        while True:
            char = encoded[index]
            index += 1
            b |= (decode_char(char) & 0x1F) << shift
            if decode_char(char) & 0x20 == 0:
                break
            shift += 5

        lat += to_signed(b)

        b, shift = 0, 0
        while True:
            char = encoded[index]
            index += 1
            b |= (decode_char(char) & 0x1F) << shift
            if decode_char(char) & 0x20 == 0:
                break
            shift += 5

        lng += to_signed(b)

        result.append((lat / 1e6, lng / 1e6))

    return result

def process_route_data(route_data):
    """Process route data and extract polyline and instructions."""
    route = json.loads(route_data)
    polyline_encoded = route['routes'][0]['sections'][0]['polyline']
    turnByTurnActions = route['routes'][0]['sections'][0]['turnByTurnActions']
    
    # Decode polyline into coordinates
    coordinates = decode_polyline(polyline_encoded)
    #print("Decoded coordinates:", coordinates)
    
    # Extract turning instructions
    instructions = extract_instructions(turnByTurnActions, coordinates)
    print("Extracted instructions:", instructions)
    
    return coordinates, instructions

def haversine_Bearing(lat1, lon1, lat2, lon2): 
    """Calculate the bearing and distance between two points on the Earth."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    initial_bearing = math.atan2(x, y)
    compass_bearing = (math.degrees(initial_bearing) + 360) % 360
    
    R = 6371  # Radius of Earth in kilometers
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(lat1) * math.cos(lat2) * 
         math.sin(dlon / 2) ** 2)
    distance = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)) * 1000  # Distance in meters

    return compass_bearing, distance

def find_turning_points(coordinates, angle_threshold=50, distance_threshold=5):
    """Identify turning points based on angle changes and distances."""
    turning_points = []
    compass_directions = []
    
    for i in range(1, len(coordinates) - 1):
        lat1, lon1 = coordinates[i - 1]
        lat2, lon2 = coordinates[i]
        lat3, lon3 = coordinates[i + 1]
        
        # Calculate angles in radians
        angle1 = math.degrees(math.atan2(lon2 - lon1, lat2 - lat1))
        angle2 = math.degrees(math.atan2(lon3 - lon2, lat3 - lat2))
        
        # Calculate distance between previous and current point
        bearing, distance_to_prev = haversine_Bearing(lat2, lon2, lat1, lon1)
        
        # Calculate change in angle
        delta_angle = abs(angle2 - angle1)
        
        idx = round(bearing / 45) % 8
        heading = compassDirection[idx]

        # Debug prints
        #print(f"Point {i}: Angle1: {angle1:.2f}, Angle2: {angle2:.2f}, Delta Angle: {delta_angle:.2f}, Distance to Prev: {distance_to_prev:.2f}")
        
        if delta_angle > angle_threshold and distance_to_prev > distance_threshold:
            turning_points.append((lat2, lon2))
            compass_directions.append(heading)  # Append the heading to the list
            print(f"Turning point identified at: {lat2}, {lon2} with angle change: {delta_angle:.2f}° and bearing: {heading}")

    return compass_directions, turning_points

def extract_instructions(turnByTurnActions, coordinates):
    """Extract instructions from route data actions."""
    instructions = []
    action_data = []

    for action in turnByTurnActions:
        action_data.append({
            "action": action['action'],
            "duration": action.get("duration"),
            "length": action.get("length"),
            "direction": action.get("direction"),
            "current_road": action.get("currentRoad", {}).get("name", [{}])[0].get("value", "N/A"),
            "next_road": action.get("nextRoad", {}).get("name", [{}])[0].get("value", "N/A")
        })
    for i, item in enumerate(action_data):
        if item["action"] == "depart":
          if len(coordinates) < 2:
              print("Not enough coordinates to calculate bearing.")
              continue  # or handle as needed

          compass, _ = haversine_Bearing(
              coordinates[0][0], coordinates[0][1], coordinates[1][0], coordinates[1][1]
          )
          idx = round(compass / 45) % 8
          next_road_name = item['next_road'] if item['next_road'] != "N/A" else "unknown road"
          instruction = f"Head {compassDirection[idx]} and continue for {item['length']}m on {next_road_name}."

        elif item["action"] == "turn":
            direction = item.get("direction")
            current_road_name = item['current_road'] if item['current_road'] != "N/A" else "unknown road"
            next_road_name = item['next_road'] if item['next_road'] != "N/A" else "unknown road"

            if current_road_name != "unknown road" and next_road_name != "unknown road":
                instruction = f"Turn {direction} from {current_road_name} onto {next_road_name}. Go for {item['length']}m."
            elif current_road_name == "unknown road":
                instruction = f"Turn {direction} onto {next_road_name}. Go straight for {item['length']}m."
            else:
                instruction = f"Continue on {current_road_name} for {item['length']}m."

        instructions.append(instruction)

    return instructions

def compareCoordinates(current_latitude, current_longitude, turning_points, instructions, threshold=20.0):
    """Compare current coordinates with turning points and return action."""
    closest_distance = float('inf')
    action = "No action at this location."
    closest_turn_point = None
    closest_idx = -1  #To track index of closest point

    for i, (turn_lat, turn_lon) in enumerate(turning_points):
        # Calculate distance to each turning point
        _, distance_to_turn = haversine_Bearing(current_latitude, current_longitude, turn_lat, turn_lon)
        #print(f"Distance to turning point {i+1} at ({turn_lat}, {turn_lon}): {distance_to_turn:.2f} meters")

        # Check if within the turning point threshold
        if distance_to_turn <= threshold:
            action = f"Action: {instructions[i]} (Time to turn!)"
            break  # Exit if within the turning point threshold
        
        # Keep track of the closest distance
        if distance_to_turn < closest_distance:
            closest_distance = distance_to_turn
            closest_turn_point = (turn_lat, turn_lon)
            closest_idx = i+1  # Save the index of the closest point

    if action == "No action at this location.":
        print(f"Closest distance to turning point {i} in {closest_distance:.2f} meters")
        print("")
    
    if closest_turn_point is not None:
        print(f"Distance to turning point {i+1} at ({turn_lat}, {turn_lon}): {distance_to_turn:.2f} meters")
        print("")
        #print(f"Closest distance to turning point {closest_idx} at ({closest_turn_point[0]}, {closest_turn_point[1]}): {closest_distance:.2f} meters")

    return action

#Connect to WiFi
connect()

#Call menu function
choice, addressName, desLatitude, desLongitude = menu()

# Main loop to fetch data and use HERE API
while True:
    try:
        while gps_serial.any():
            data = gps_serial.read()
            for byte in data:
                stat = my_gps.update(chr(byte))
                if stat is not None:
                    if my_gps.satellites_in_use > 0:  # Indicates a valid fix
                        latitude = my_gps.latitude[0] + my_gps.latitude[1] / 60
                        longitude = my_gps.longitude[0] + my_gps.longitude[1] / 60
                        latitude = round(latitude, 6)
                        longitude = round(longitude, 6)

                        # Print parsed GPS data
                        local_time = localtime()  # Get local time
                        print(f"Local Time (SG): {local_time[3]:02d}:{local_time[4]:02d}:{local_time[5]:02d}")
                        print('Date:', my_gps.date_string('long'))
                        print('Latitude:', latitude)
                        print('Longitude:', longitude)
                        #print('Altitude:', my_gps.altitude)
                        #print('Satellites in use:', my_gps.satellites_in_use)
                        print('Horizontal Dilution of Precision:', my_gps.hdop)
                        
                        if not route_printed:
                            # Handle destination queries
                            if choice == "Address" and addressName:
                                destination = searchAddress(latitude, longitude, addressName)
                                if destination:
                                    print("Address: ", destination)
                                else:
                                    print("Searching for address...")
                            elif choice == "Coordinates" and desLatitude and desLongitude:
                                destination = desRevGeoCode(desLatitude, desLongitude)
                                if destination:
                                    print("Destination: ", destination)
                                else:
                                    print("Error getting rev geocode destination")
                        
                            location = revGeoCode(latitude, longitude)
                        
                            if location:
                                print("Location: ", location)
                                if not calibrate:
                                    choiceCoordinates = input(f"Is {location} your starting location? ")
                                    
                                    while choiceCoordinates != "Yes" :
                                            calibrate = True
                                            latitude = input(f"Please input your current latitude: ")
                                            longitude = input(f"Please input your current longitude: ")
                                            location = revGeoCode(latitude, longitude)
                                            choiceCoordinates = input(f"Is {location} your starting location? ")
     
                            else:
                                print("Error getting rev geocode location")
                        
                            turnByTurnActions = route(latitude, longitude, desLatitude, desLongitude)

                            if turnByTurnActions:  # Check if valid actions were returned
                                # Call directions with the entire route data for debugging                         
                                coordinates, instructions = process_route_data(turnByTurnActions)  # Adjust this line
                                compass_directions, turning_points = find_turning_points(coordinates)
                                for instruction in instructions:  # Print each instruction
                                    #print("Action: ", instruction)
                                    route_printed = True
                                
                            # Call the compareCoordinates function with inputs and print the result
                            action = compareCoordinates(latitude, longitude, turning_points, instructions)
                            print(action)
                        
                        time.sleep(1)
                    
                    else:
                        print("Waiting for GPS fix...")
    except Exception as e:
        print(f"An error occurred: {e}")