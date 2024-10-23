from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

#Initialise current coordinates
current_coordinates = {
    'start_latitude': None,
    'start_longitude': None,
    'end_latitude': None,
    'end_longitude': None,
    'azimuthAngle': None,
    'userDirection': None
}

location_coordinates ={
    'start_latitude': None,
    'start_longitude': None,
    'end_latitude': None,
    'end_longitude': None,
}

# Initialize passing arrays
waypoints = []
instructions = []
intendedDir = []
bearingWLocations = []
turnAngle = []

#Initialise int
currentStep = 0

@app.route('/')
def map():
    return render_template('index.html')

@app.route('/coordinates', methods=['GET']) #For client to get current_coordinates
def get_coordinates():
    return jsonify(current_coordinates) #Also contains azimuth and user direction


@app.route('/receive_coordinates', methods=['POST'])    #Esp32 sends to flask
def receive_coordinates():
    global current_coordinates
    data = request.get_json()

    if not data or 'current_latitude' not in data or 'current_longitude' not in data:
        return jsonify({"error": "Invalid input"}), 400  # Bad request

    try:
        current_latitude = data['current_latitude']
        current_longitude = data['current_longitude']
        userAzimuth = data['azimuth']
        userDirection = data['direction']
        
        # Update the current coordinates
        current_coordinates['start_latitude'] = current_latitude
        current_coordinates['start_longitude'] = current_longitude
        current_coordinates['azimuthAngle'] = userAzimuth
        current_coordinates['userDirection'] = userDirection
        
        #print(f"Received coordinates: {current_latitude}, {current_longitude}")    #debugging line
        return jsonify({'status': 'success', 'message': 'Current coordinates received.', 'updated_coordinates': current_coordinates}), 200
    except (KeyError, ValueError) as e:
        return jsonify({'status': 'error', 'message': f'Error processing data: {str(e)}'}), 400


@app.route('/update_coordinates', methods=['POST']) #JS sends start and end to flask
def update_coordinates():
    global location_coordinates
    data = request.get_json()
    
    if data is not None:
        try:
            # Extract coordinates from the incoming data
            start_lat = float(data['start_latitude'])
            start_lon = float(data['start_longitude'])
            destination_latitude = float(data['end_latitude'])
            destination_longitude = float(data['end_longitude'])
            
            # Update the global coordinates
            location_coordinates['start_latitude'] = start_lat
            location_coordinates['start_longitude'] = start_lon
            location_coordinates['end_latitude'] = destination_latitude
            location_coordinates['end_longitude'] = destination_longitude

            print(f"Updated Coordinates: Start({start_lat}, {start_lon}), Destination({destination_latitude}, {destination_longitude})")
            return jsonify({'status': 'success', 'message': 'Coordinates updated.'}), 200
        except (KeyError, ValueError) as e:
            return jsonify({'status': 'error', 'message': f'Error processing data: {str(e)}'}), 400
    
    return jsonify({'status': 'error', 'message': 'No data received.'}), 400

@app.route('/get_coordinates', methods=['GET'])
def get_coordinates_for_esp32():
    """Endpoint for ESP32 to get coordinates."""
    return jsonify(location_coordinates)  # Return the current coordinate


@app.route('/update_waypoints', methods=['POST'])
def update_waypoints():
    global waypoints
    data = request.get_json()
    
    if data is not None:
        try:
            waypoints = data.get('waypoints', [])  # Update the global waypoints variable
            print(f"Received waypoints: {waypoints}")  # Log received waypoints
            return jsonify({'status': 'success', 'message': 'Coordinates updated.'}), 200
        except (KeyError, ValueError) as e:
            return jsonify({'status': 'error', 'message': f'Error processing data: {str(e)}'}), 400
    
    return jsonify({'status': 'error', 'message': 'No data received.'}), 400

@app.route('/get_waypoints', methods=['GET'])
def get_waypoints_for_esp32():
    """Endpoint for ESP32 to get waypoint coordinates."""
    return jsonify(waypoints)  # Return the turning points


@app.route('/update_instructions', methods=['POST'])
def update_instructions():
    global instructions
    data = request.get_json()
    
    if data is not None:
        try:
            instructions = data.get('instructions', [])  # Update the global waypoints variable
            print(f"Received instructions: {instructions}")  # Log received waypoints
            return jsonify({'status': 'success', 'message': 'Coordinates updated.'}), 200
        except (KeyError, ValueError) as e:
            return jsonify({'status': 'error', 'message': f'Error processing data: {str(e)}'}), 400
    
    return jsonify({'status': 'error', 'message': 'No data received.'}), 400

@app.route('/get_instructions', methods=['GET'])
def get_instructions_for_esp32():
    """Endpoint for ESP32 to get instructions"""
    return jsonify(instructions)  # Return the instructions

@app.route('/update_turnAngle', methods=['POST'])
def update_turnAngle():
    global turnAngle
    data = request.get_json()
    
    if data is not None:
        try:
            turnAngle = data.get('turnAngle', [])  # Update the global waypoints variable
            print(f"Received turnAngle: {turnAngle}")  # Log received waypoints
            return jsonify({'status': 'success', 'message': 'Coordinates updated.'}), 200
        except (KeyError, ValueError) as e:
            return jsonify({'status': 'error', 'message': f'Error processing data: {str(e)}'}), 400
    
    return jsonify({'status': 'error', 'message': 'No data received.'}), 400

@app.route('/get_turnAngle', methods=['GET'])
def get_turnAngle_for_esp32():
    """Endpoint for ESP32 to get turnAngle"""
    return jsonify(turnAngle)  # Return the instructions


@app.route('/update_bearingWLocations', methods=['POST'])
def update_bearingWLocations():
    global bearingWLocations
    data = request.get_json()
    
    if data is not None:
        try:
            bearingWLocations = data.get('bearingWLocations', [])  # Update the global waypoints variable
            print(f"Received bearingWLocations: {bearingWLocations}")  # Log received waypoints
            return jsonify({'status': 'success', 'message': 'Coordinates updated.'}), 200
        except (KeyError, ValueError) as e:
            return jsonify({'status': 'error', 'message': f'Error processing data: {str(e)}'}), 400
    
    return jsonify({'status': 'error', 'message': 'No data received.'}), 400

@app.route('/get_bearingWLocations', methods=['GET'])
def get_bearingWLocations_for_esp32():
    """Endpoint for ESP32 to get bearingWLocations"""
    return jsonify(bearingWLocations)  # Return the instructions


@app.route('/currentStep', methods=['GET'])
def get_currentStep():
    return jsonify(currentStep)

@app.route('/receive_currentStep', methods=['POST'])
def receive_currentStep():
    global currentStep
    data = request.get_json()

    if not data or 'currentStep' not in data:
        return jsonify({"error": "Invalid input"}), 400  # Bad request

    try:
        currentStep = data['currentStep']

        #print(f"Received coordinates: {current_latitude}, {current_longitude}")    #debugging line
        return jsonify({'status': 'success', 'message': 'Current coordinates received.', 'updated_coordinates': current_coordinates}), 200
    except (KeyError, ValueError) as e:
        return jsonify({'status': 'error', 'message': f'Error processing data: {str(e)}'}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5501, debug=True)
