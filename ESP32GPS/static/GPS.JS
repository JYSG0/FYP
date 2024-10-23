let azimuth = null;
let userDirection = null;
let currentStep = null;
let currentLat = null;
let currentLon = null;
let routeData = null;
let midRouteData = null;

// Global variable to store the current active tab
let currentTab = 'fullRoute'; // Default to 'fullRoute'

let start = [];  //Starting position coordinates
let end = [];    //Destination position coordinates
let current = [];
let viewRoutes = false; //Check to see if button is pressed
let midRoute = false;   //Check to determine if user joins midroute

//Initialize an empty array to store waypoints
let waypoints = [];
let myWaypoints = [];
let stepBearing = [];
//Array to store waypoint markers
let waypointMarkers = [];
let myWaypointMarkers = [];

//Initialize an empty array to store instructions
let directions = [];
let myDirections = [];
let bearingWLocations = [];
let turnAngle = [];

//Finding closest waypoint if starting midRoute
let currentBearings = [];
let currentDistances = [];

//Store route data for full route and midroute
let route = null;
let myRoute = null;

//Boolean checks to track active search box
let isCurrentSearchBox = false;
let isDestinationSearchBox = false;

//Create variables to store the start and destination markers
let startMarker = null;
let destinationMarker = null;

//Compass variables to identify bearings and direction
let compassDirection = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
let compassRanges = [];
let rotation;

//Travel type choice
let travelType;

//Mapbox Api key
mapboxgl.accessToken = 'pk.eyJ1IjoiYW1hbmRheWFwIiwiYSI6ImNtMXFsOXh0aDAxYjYyaW9lbTkydjRhNTEifQ.q_7I3PMALzQlmJ62VfB76w';

//Initialise map
const map = new mapboxgl.Map({
    container: 'map', //container ID
    center: [103.777353, 1.309636], //starting position [lng, lat]
    zoom: 16, //starting zoom
    pitch: 30   //Set at angle to see cool 3D buildings
});

//Add full view control
map.addControl(new mapboxgl.FullscreenControl(), 'bottom-left');

//Add zooms and compass
const compassControl = new mapboxgl.NavigationControl();
map.addControl(compassControl, 'bottom-right');

//Orientate map
map.on('rotate', function() {
    rotation = map.getBearing(); // Get current map bearing (rotation)

    // Rotate the cardinal directions around their center
    const cardinalDirections = document.querySelector('#cardinal-directions');
    cardinalDirections.style.transform = `rotate(${rotation}deg)`;
});

//Function to fetch coordinates from the server
async function fetchCoordinates() {
    try {
        const response = await fetch('/coordinates');  //GET request to your Flask route
        if (!response.ok) {
            const errorData = await response.json();
            console.error('Error fetching coordinates:', errorData.message);
            return;
        }

        const data = await response.json();
        console.log('Coordinates received:', data);
        
        //Assuming you want to use the received coordinates
        current = [data.start_longitude, data.start_latitude];  //Adjust according to your data structure
        azimuth = data.azimuthAngle;
        userDirection = data.userDirection;
        console.log('Current location:', current);
        console.log('Angle', azimuth);
        console.log('Rotation', rotation);

        //Update the triangle's position and orientation
        currentLat = current[1];
        currentLon = current[0];

        const triangle = document.getElementById('triangle');
        if (triangle) {
            let updatedAzimuth;
            updatedAzimuth = azimuth + rotation;
            console.log(updatedAzimuth);
            if (updatedAzimuth > 360) {
                updatedAzimuth = updatedAzimuth - 360;
            }
            else{
                updatedAzimuth = azimuth;
            }
            updateTrianglePosition(current, updatedAzimuth); //Update location and angle
        } 
        else {
            console.error('Triangle element not found');
        }
        
    } 
    catch (error) {
        console.error('Fetch error:', error);
    }
}

// Set the initial rotations based on the initial map bearing
const initialRotation = map.getBearing();
document.querySelector('#cardinal-directions').style.transform = `rotate(${initialRotation}deg)`;

//Function to geocode coordinates
const coordinatesGeocoder = function (query) {
    const matches = query.match(/^[ ]*(?:Lat: )?(-?\d+\.?\d*)[, ]+(?:Lng: )?(-?\d+\.?\d*)[ ]*$/i);
    if (!matches) return null;

    function coordinateFeature(lng, lat) {
        return {
            center: [lng, lat],
            geometry: {
                type: 'Point',
                coordinates: [lng, lat]
            },
            place_name: 'Lat: ' + lat + ' Lng: ' + lng,
            place_type: ['coordinate'],
            properties: {},
            type: 'Feature'
        };
    }

    const coord1 = Number(matches[1]);
    const coord2 = Number(matches[2]);
    const geocodes = [];

    if (coord1 < -90 || coord1 > 90) {
        geocodes.push(coordinateFeature(coord1, coord2));
    }
    if (coord2 < -90 || coord2 > 90) {
        geocodes.push(coordinateFeature(coord2, coord1));
    }
    if (geocodes.length === 0) {
        geocodes.push(coordinateFeature(coord1, coord2));
        geocodes.push(coordinateFeature(coord2, coord1));
    }

    return geocodes;
};

//Reverse geocode to get place name
function reverseGeocode(coords) {
    console.log('Reverse geocoding for coordinates:', coords); //Debug: log coordinates
    const url = `https://api.mapbox.com/geocoding/v5/mapbox.places/${coords[0]},${coords[1]}.json?access_token=${mapboxgl.accessToken}`;

    console.log('Rev-geocoding URL:', url); //Debug: log URL

    fetch(url)
        .then((response) => {
            if (!response.ok) {
                throw new Error('Network response was not ok: ' + response.statusText);
            }
            return response.json();
        })
        .then((data) => {
            console.log('Reverse geocode response:', data); //Debug: log response
            const placeName = data.features[0]?.place_name || 'Unknown location';
            if (isDestinationSearchBox) {
                geocoderDestination.setInput(placeName); //Update search box with the new location
                console.log('Selected Destination:', placeName);
                console.log('Destination Coordinates:', end); //Log end coordinates
            } else if (isCurrentSearchBox) {
                geocoderStart.setInput(placeName); //Update search box with the new location
                console.log('Selected Current Location:', placeName);
                console.log('Starting Location Coordinates:', start); //Log start coordinates
            }
            updateMarkers(); //Update markers for current location and destination
        })
        .catch((error) => {
            console.error('Error with reverse geocoding:', error);
        });
}

//Add event listener to recentre map when the button is clicked
document.getElementById('recentre-button').addEventListener('click', () => {
    recentreMapToCurrentLocation();  //Assuming this function is defined
});

//Function to recentre the map to the current coordinates
function recentreMapToCurrentLocation() {
    //Check if current location is available
    if (current && current.length === 2) {
        const targetLocation = [current[0], current[1]];
        //recentre the map using Mapbox's flyTo method
        map.flyTo({
            center:targetLocation,
            essential: true,
            zoom: 17,
            speed: 2.5,
            pitch:30,
            curve:1,
            easing: (t) => t
        })
        
        console.log(`Map recentred to current location: (${currentLatitude}, ${currentLongitude})`);
    } else {
        console.error('Current location not available to recentre the map.');
    }
}

//==================================== PURPLE CURRENT LOCATION ====================================//
//Show pulsing marker to indicate current position
const size = 100;

//This implements `StyleImageInterface` to draw a pulsing dot icon on the map.
const pulsingDot = {
    width: size,
    height: size,
    data: new Uint8Array(size * size * 4),

    onAdd: function () {
        const canvas = document.createElement('canvas');
        canvas.width = this.width;
        canvas.height = this.height;
        this.context = canvas.getContext('2d');
    },

    render: function () {
        const duration = 1000;
        const t = (performance.now() % duration) / duration;

        const radius = (size / 2) * 0.3;
        const outerRadius = (size / 2) * 0.7 * t + radius;
        const context = this.context;

        //Draw the outer circle.
        context.clearRect(0, 0, this.width, this.height);
        context.beginPath();
        context.arc(this.width / 2, this.height / 2, outerRadius, 0, Math.PI * 2);
        context.fillStyle = `rgba(174, 48, 240, ${1 - t})`;
        context.fill();

        //Draw the inner circle.
        context.beginPath();
        context.arc(this.width / 2, this.height / 2, radius, 0, Math.PI * 2);
        context.fillStyle = 'rgba(174, 48, 240, 1)';
        context.strokeStyle = 'white';
        context.lineWidth = 2 + 4 * (1 - t);
        context.fill();
        context.stroke();

        //Update this image's data with data from the canvas.
        this.data = context.getImageData(0, 0, this.width, this.height).data;

        map.triggerRepaint();

        return true;
    }
};

//Triangle
function updateTrianglePosition(location, angle) {
    console.log('Updating triangle position')
    const triangle = document.getElementById('triangle');

    //Rotate triangle to point according to azimuth
    triangle.style.transform = `translate(-50%, -100%) rotate(${angle}deg)`; //Adjust position and rotation
}
//================================================================================================//

//Compass directions
function updateCompassDirection(azimuth, rotation){
    const updatedAzimuth = azimuth + rotation;
    if (updatedAzimuth > 360) {
        azimuth = updatedAzimuth - 360;
    }

    return azimuth;
}

//==================================== RED DESTINATION MARKER ====================================//
//Add the geocoders for destination
const geocoderDestination = new MapboxGeocoder({
    accessToken: mapboxgl.accessToken,
    localGeocoder: coordinatesGeocoder,
    zoom: 17,
    placeholder: 'Enter destination',
    mapboxgl: mapboxgl,
    reverseGeocode: true,
    marker: false
});

//Check if geocoded correctly
console.log(geocoderDestination);
//Add search box onto map
document.getElementById('geocoderDestination').appendChild(geocoderDestination.onAdd(map));
//Add focus event listeners
document.getElementById('geocoderDestination').querySelector('input').addEventListener('focus', () => {
    isCurrentSearchBox = false;
    isDestinationSearchBox = true;
});
//================================================================================================//

//===================================== BLUE STARTING MARKER =====================================//
const geocoderStart = new MapboxGeocoder({
    accessToken: mapboxgl.accessToken,
    localGeocoder: coordinatesGeocoder,
    zoom: 17,
    placeholder: 'Using current location as starting',
    mapboxgl: mapboxgl,
    reverseGeocode: true,
    marker: false
});

//Check if geocoded correctly
console.log(geocoderStart);
//Add search box onto the map
document.getElementById('geocoderStart').appendChild(geocoderStart.onAdd(map));
//Set focus event listeners
document.getElementById('geocoderStart').querySelector('input').addEventListener('focus', () => {
    isCurrentSearchBox = true;
    isDestinationSearchBox = false;
});
//================================================================================================//

//=====================================  DIRECTIONS BUTTON =======================================//
const viewRoutesButton = document.getElementById('view-route');
if (viewRoutesButton) {
    viewRoutesButton.addEventListener('click', () => {

        isCurrentSearchBox = false;
        isDestinationSearchBox = false;

        if (start.length === 0) {
            start = [...current];
            midRoute = false
        } else if (end.length === 0) {
            end = [...current];
            midRoute = false
        }
        else {
            midRoute = true
        }

        const [startLon, startLat] = start;
        const [endLon, endLat] = end;

        console.log(`Start route from (${startLat}, ${startLon}) to (${endLat}, ${endLon})`);
        
        viewRoutes = true;

        const dataToSend = {
            start_latitude: startLat,
            start_longitude: startLon,
            end_latitude: endLat,
            end_longitude: endLon
        };

        fetch('/update_coordinates', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(dataToSend)
        })
        .then(response => response.json())
        .then(data => {
            console.log('Coordinates successfully sent to server:', data);
        })
        .catch((error) => {
            console.error('Error sending coordinates to server:', error);
        });

        updateMarkers();
        getRoute(startLat, startLon, endLat, endLon, travelType);
        console.log(`Midroute: ${midRoute}`);
        if (midRoute){
            reRoute(currentLat, currentLon, endLat, endLon, travelType);
        }
    });
} 
else {
    console.error('View Routes button not found in the DOM');
}
//===============================================================================================//

//====================================== START ROUTE BUTTON =====================================//
let routeActive = false;
const startRouteButton = document.getElementById('start-route');
const clearMarkersButton = document.getElementById('clear-markers');
const shuffleInstrButton = document.getElementById('shuffle-instr');
shuffleInstrButton.disabled = true;

if (startRouteButton && clearMarkersButton) {
    startRouteButton.addEventListener('click', () => {
        if (!routeActive) {
            startRouteButton.value = "Stop Route";
            routeActive = true; //Route has started
            
            //Enable the shuffle instructions style button when route is active
            shuffleInstrButton.disabled = false;
            //Disable the clear markers button when the route is active
            clearMarkersButton.disabled = true;

            //Send waypoints to server when starting the route
            if (midRoute){
                sendInstructionsToServer(myDirections);
                sendWaypointsToServer(myWaypoints);
            }
            else{
                sendInstructionsToServer(directions);
                sendWaypointsToServer(waypoints);
            }
            sendRouteBearingToServer(bearingWLocations);
            sendTurnAngleToServer(turnAngle);
            fetchCurrentStep();
        } else {
            startRouteButton.value = "Start Route";
            routeActive = false;
            start = [];
            end = [];

            clearMarkersButton.disabled = false;

            shuffleInstrButton.disabled = true;

            //Link to wheels or something to stop the route
        }
    });
}
else {
    console.error('Start Route button not found in the DOM');
}
//===============================================================================================//

//===================================== TRAVEL TYPE BUTTONS =====================================//
const travelTypeCarButton = document.getElementById('travel-type-car');
const travelTypeWalkingButton = document.getElementById('travel-type-walking');

travelTypeCarButton.addEventListener('click', () => {
    travelTypeCarButton.classList.add('active');
    travelTypeWalkingButton.classList.remove('active');
    travelType = 'driving';
});
travelTypeWalkingButton.addEventListener('click', () => {
    travelTypeCarButton.classList.remove('active');
    travelTypeWalkingButton.classList.add('active');
    travelType = 'walking'
});
//===============================================================================================//

//==================================== CLEAR MARKERS BUTTON =====================================//
clearMarkersButton.addEventListener('click', () =>{
    if (!routeActive){
        //Clear markers on map
        clearMarkers();

        start = [];
        end = [];

        updateMarkers();

        //Clear the search boxes for start and destination
        if (geocoderStart) {
            geocoderStart.value = '';  //Clear start search box
        }
        
        if (geocoderDestination) {
            geocoderDestination.value = '';  //Clear destination search box
        }

        //Clear routes on map
        clearPreviousRoutes();
        
        //Get all layers currently on the map mid route
        map.getStyle().layers.forEach(layer => {
            //Check if the layer ID starts with 'route-' to identify both main and alternative routes
            if (layer.id.startsWith('myRoute-')) {
                map.removeLayer(layer.id); //Remove the layer
                map.removeSource(layer.id); //Remove the corresponding source
            }
        });

        directions = [];
        myDirections = [];

        isCurrentSearchBox = false;
        isDestinationSearchBox = false;

        console.log('Markers and routes cleared');
    }
    else{
        console.log('Start Route or Clear Markers button not found')
    }
})
//===============================================================================================//

//================================= TOGGLE INSTRUCTIONS BUTTON ==================================//
document.addEventListener("DOMContentLoaded", () => {
    //Get the toggle button and the instructions container
    const toggleButton = document.getElementById('toggle-instructions');
    const instructionsContainer = document.getElementById('instructions');

    let showInstructions = false; //To track the visibility state

    toggleButton.addEventListener('click', () => {
        showInstructions = !showInstructions; //Toggle the state

        if (showInstructions) {
            //Show instructions
            instructionsContainer.style.display = 'block'; //Set display to block
            console.log('Instructions shown'); //Optional: Log to console
        } else {
            //Hide instructions
            instructionsContainer.style.display = 'none'; //Set display to none
            console.log('Instructions hidden'); //Optional: Log to console
        }
    });
});
//===============================================================================================//

//============================== SHUFFLE INSTRUCTIONS STYLE BUTTON ==============================//
document.getElementById('shuffle-instr').addEventListener('click', function() {
    // Toggle active class for shuffle button
    this.classList.toggle('active');

    // Get instructions container and toggle active state
    var instructionsCont = document.getElementById('instructions');
    instructionsCont.classList.toggle('active');
    
    // Hide the full route and mid route tabs when shuffle is active
    var fullRouteButton = document.querySelector('button[onclick="showTab(\'fullRoute\')"]');
    var midRouteButton = document.querySelector('button[onclick="showTab(\'midRoute\')"]');
    
    // Elements for instructions list
    var instructionsListFull = document.getElementById('instructions-list-full');
    var instructionsListMid = document.getElementById('instructions-list-mid'); // Placeholder for shuffled instructions

    // Track current step in the shuffled instructions
    let currentStep = 0;

    // If the shuffle button is active, hide the tabs and always show mid-route instructions
    if (this.classList.contains('active') && routeActive){
        fullRouteButton.style.display = 'none'; // Hide Full Route tab
        midRouteButton.style.display = 'none'; // Hide Mid Route tab
        
        // Clear the instructions list and replace with mid-route shuffled instructions
        instructionsListFull.style.display = 'none'; // Hide the full route list
        instructionsListMid.style.display = 'block'; // Show the mid-route shuffled instruction display

        // Clear the full route duration
        const fullRouteDuration = document.getElementById('full-route-duration');
        const midRouteDuration = document.getElementById('mid-route-duration');
        fullRouteDuration.innerHTML = ''; // Clear the full trip duration
        midRouteDuration.innerHTML = ''; // Clear mid route duration

        if (midRoute){
            // Show the mid-route shuffled instructions regardless of midRoute flag
            showCurrentInstruction(myDirections, currentStep, myWaypoints, currentLat, currentLon);
            
            // Copy mid-route instructions to full-route list
            instructionsListFull.innerHTML = instructionsListMid.innerHTML;
            instructionsListFull.style.display = 'block'; // Show the copied instructions
        }
        else{
            showCurrentInstruction(directions, currentStep, waypoints);
            // Copy mid-route instructions to full-route list
            instructionsListFull.innerHTML = instructionsListMid.innerHTML;
            instructionsListFull.style.display = 'block'; // Show the copied instructions
            console.log('Full route instructions not applicable');
        }
        
        console.log('Shuffle mode ON: Displaying mid-route shuffled instructions.');
    } 
    else {
        // Shuffle mode off, show the Full Route and Mid Route buttons again
        fullRouteButton.style.display = 'inline-block';
        midRouteButton.style.display = 'inline-block';
        
        // Switch back to full route and mid route instructions based on the default state
        instructionsListFull.style.display = 'block';
        instructionsListMid.style.display = 'block';
        
        instructions(routeData); // Restore full route instructions
        myInstructions(midRouteData); // Restore mid-route instructions

        console.log('Shuffle mode OFF: Restoring full route and mid-route instructions.');
    }
});
//===============================================================================================//

//Update markers for starting and destination locations
const updateMarkers = () => {
    //Remove existing markers if they exist
    if (startMarker) {
        startMarker.remove();
    }
    if (destinationMarker) {
        destinationMarker.remove();
    }

    //Create new start marker if a start location exists
    if (start.length) {
        startMarker = new mapboxgl.Marker({ color: '#3b9ddd' }) //Blue marker for start
            .setLngLat(start)
            .addTo(map);
    }

    //Create new destination marker if an end location exists
    if (end.length) {
        destinationMarker = new mapboxgl.Marker({ color: 'red' }) //Red marker for destination
            .setLngLat(end)
            .addTo(map);
    }
};

//Listen for the result event from destination geocoder
geocoderDestination.on('result', (e) => {
    const selectedFeature = e.result;
    end = selectedFeature.geometry.coordinates;
    console.log('Selected Destination:', selectedFeature.place_name);
    console.log('Destination Coordinates:', end);
    updateMarkers(); //Ensure markers are updated immediately
});

//Listen for the result event from current location geocoder
geocoderStart.on('result', (e) => {
    const selectedFeature = e.result;
    start = selectedFeature.geometry.coordinates; //Update start coordinates
    console.log('Selected Current Location:', selectedFeature.place_name);
    console.log('Current Location Coordinates:', start);
    updateMarkers(); //Ensure markers are updated immediately
});

//Function to get route from starting to end
async function getRoute(startLat, startLon, endLat, endLon, travelType) {
    console.log('Getting Route w travel type: ', travelType);
    const query = await fetch(
        `https://api.mapbox.com/directions/v5/mapbox/${travelType}/${startLon},${startLat};${endLon},${endLat}?steps=true&geometries=geojson&overview=full&alternatives=true&waypoints_per_route=true&access_token=${mapboxgl.accessToken}`,
        { method: 'GET' }
    );
    const json = await query.json();
    const routes = json.routes;
    console.log(routes)

    //Clear previous routes and markers before adding new ones
    clearPreviousRoutes();
    clearMarkers();
    
    waypoints = []; //Reset waypoints before extracting new ones

    //Loop through all the routes returned (main route + alternatives)
    routes.forEach((data, index) => {
        route = data.geometry.coordinates;

        const geojson = {
            type: 'Feature',
            properties: {},
            geometry: {
                type: 'LineString',
                coordinates: route
            }
        };

        //Create a unique ID for each route layer (main route is `route-0`, alternatives `route-1`, `route-2`, etc.)
        const routeId = `route-${index}`;

        //If the route (including alternatives) already exists on the map, reset it using setData
        if (map.getSource(routeId)) {
            map.getSource(routeId).setData(geojson);
        } else {
            //Add the route to the map if it doesn't exist yet
            map.addSource(routeId, {
                type: 'geojson',
                data: geojson
            });

            //Add the line layer for the route (different colors for the main route and alternatives)
            map.addLayer({
                id: routeId,
                type: 'line',
                source: routeId,
                layout: {
                    'line-join': 'round',
                    'line-cap': 'round'
                },
                paint: {
                    'line-color': index === 0 ? '#3b9ddd' : '#FF7F50', //First route (main) is blue, others (alternatives) are gray
                    'line-width': 5
                }
            });
        }
        routeData = data;
        //Add route instructions and extract waypoints
        instructions(data); //Function to display or handle route instructions
        waypoints = extractWaypoints(data); //Extract waypoints from the route data
        if (!midRoute){ //Current location is starting point
            reRoute(startLat, startLon, endLat, endLon);
        }
    });

    showTab('fullRoute');
    console.log('Routes and waypoints updated');
}

//Function to get current location to end
async function reRoute(currentLat, currentLon, endLat, endLon, travelType) {
    const query = await fetch(
        `https://api.mapbox.com/directions/v5/mapbox/${travelType}/${currentLon},${currentLat};${endLon},${endLat}?steps=true&geometries=geojson&overview=full&alternatives=true&waypoints_per_route=true&access_token=${mapboxgl.accessToken}`,
        { method: 'GET' }
    );
    const json = await query.json();
    const routes = json.routes;
    console.log(routes)
    
    myWaypoints = []; //Reset waypoints before extracting new ones

    //Loop through all the routes returned (main route + alternatives)
    routes.forEach((myData, index) => {
        myRoute = myData.geometry.coordinates;

        const geojson = {
            type: 'Feature',
            properties: {},
            geometry: {
                type: 'LineString',
                coordinates: myRoute
            }
        };

        //Create a unique ID for each route layer (main route is `route-0`, alternatives `route-1`, `route-2`, etc.)
        const routeId = `myRoute-${index}`;

        //If the route (including alternatives) already exists on the map, reset it using setData
        if (map.getSource(routeId)) {
            map.getSource(routeId).setData(geojson);
        } 
        else {
            //Add the route to the map if it doesn't exist yet
            map.addSource(routeId, {
                type: 'geojson',
                data: geojson
            });

            //Add the line layer for the route (different colors for the main route and alternatives)
            map.addLayer({
                id: routeId,
                type: 'line',
                source: routeId,
                layout: {
                    'line-join': 'round',
                    'line-cap': 'round'
                },
                paint: {
                    'line-color': index === 0 ? '#ae30f0' : '#FF7F50', //First route (main) is purple, others (alternatives) are gray
                    'line-width': 5
                }
            });
        }

        //console.log(myRoute)  //Debugging line
        
        midRouteData = myData;
        //Add route instructions and extract waypoints
        myInstructions(myData); //Function to display or handle route instructions
        myWaypoints = extractMyWaypoints(myData); //Extract waypoints from the route data
        calculateTurningAngle(myData);
        routeBearing(myRoute);
    });

    console.log('Routes and waypoints updated');
}

//Function to clear routes on map
function clearPreviousRoutes() {
    //Get all layers currently on the map full route
    map.getStyle().layers.forEach(layer => {
        //Check if the layer ID starts with 'route-' to identify both main and alternative routes
        if (layer.id.startsWith('route-')) {
            map.removeLayer(layer.id); //Remove the layer
            map.removeSource(layer.id); //Remove the corresponding source
        }
    });

    console.log('Previous main and alternative routes cleared');
}

//Function to clear waypoint markers on map
function clearMarkers() {
    //Remove all waypoint markers from the map
    waypointMarkers.forEach(marker => marker.remove());
    waypointMarkers = []; //Reset the marker array
    myWaypointMarkers.forEach(marker => marker.remove());
    myWaypointMarkers = [];
    console.log('All markers cleared');
}

// Function to show the correct tab and hide others
function showTab(tabName) {
    // Hide all tab content
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(tabContent => {
        tabContent.style.display = 'none'; // Hide each tab content
    });

    // Show the selected tab content
    const activeTab = document.getElementById(tabName);
    if (activeTab) {
        activeTab.style.display = 'block'; // Show the active tab content
    }

    // Update the currentTab variable to the selected tab
    currentTab = tabName;

    // Update the tab button styles
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(tabButton => {
        if (tabButton.getAttribute('onclick').includes(tabName)) {
            // Set styles for the active tab button
            tabButton.classList.add('tab-button-active');
            tabButton.classList.remove('tab-button-inactive');
        } else {
            // Set styles for the inactive tab button
            tabButton.classList.remove('tab-button-active');
            tabButton.classList.add('tab-button-inactive');
        }
    });

    console.log('Current Tab:', currentTab); // This will print the current active tab name
}

//Route full route
function instructions(data, shuffleMode = false) {
    //Get the instructions list container for Full Route
    const instructionsList = document.getElementById('instructions-list-full');
    const steps = data.legs[0].steps;

    //Clear previous instructions
    instructionsList.innerHTML = '';

    //Reset directions array
    directions = [];

    //Add instructions to the list
    for (const step of steps) {
        const li = document.createElement('li');
        li.textContent = step.maneuver.instruction; //Add instruction text
        instructionsList.appendChild(li); //Append to the full route list
        directions.push(step.maneuver.instruction);
    }

    //Update the Full Route trip duration
    const fullRouteDuration = document.getElementById('full-route-duration');
    if (!shuffleMode){
        fullRouteDuration.innerHTML = `<strong>Full trip duration: ${Math.floor(data.duration / 60)} min 🚗 </strong>`;
    }
    else{
        fullRouteDuration.innerHTML = ''
    }
    console.log(`Full instructions updated.`);
}

//Route user to route
function myInstructions(data, shuffleMode = false) {
    //Get the mid route instructions list container
    const myInstructionsListMid = document.getElementById('instructions-list-mid');
    if (!myInstructionsListMid) {
        console.error("Element 'instructions-list-mid' not found");
        return; //Exit the function if the element is not found
    }
    
    console.log('instructions-list-mid');
    const steps = data.legs[0].steps;

    //Clear previous instructions
    myInstructionsListMid.innerHTML = '';

    //Reset directions array
    myDirections = [];

    //Add instructions to the mid route list
    for (const step of steps) {
        const liMid = document.createElement('li');
        liMid.textContent = step.maneuver.instruction; //Add instruction text
        myInstructionsListMid.appendChild(liMid);  //Append to mid route list
        myDirections.push(step.maneuver.instruction);
    }

    //Update the Full Route trip duration
    const midRouteDuration = document.getElementById('mid-route-duration');
    if (!shuffleMode){
        midRouteDuration.innerHTML = `<strong>Mid trip duration: ${Math.floor(data.duration / 60)} min 🛣️ </strong>`;
    }
    else{
        midRouteDuration.innerHTML = ''
    }
    console.log(`Mid instructions updated.`);
}

function sendInstructionsToServer(directions){
    fetch('/update_instructions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            instructions: directions //Sending the array of waypoint coordinates
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('instructions successfully sent to server:', data);
    })
    .catch((error) => {
        console.error('Error sending instructions to server:', error);
    });
}

//Function to extract waypoints from the route data
function extractWaypoints(data) {
    //Clear the previous waypoints array
    waypoints = [];

    //Remove previous markers from the map
    waypointMarkers.forEach(marker => marker.remove());
    waypointMarkers = []; //Clear the array of markers

    const steps = data.legs[0].steps;

    //Loop through each step to extract waypoints
    steps.forEach(step => {
        const waypointCoordinates = step.maneuver.location; //[longitude, latitude]

        //Add the waypoint to the array
        waypoints.push(waypointCoordinates);

        const marker = document.createElement('div');
        marker.className = 'waypoint-marker'; //Add a class for styling

        //Create the marker and set its location
        const waypointMarker = new mapboxgl.Marker(marker)
            .setLngLat([waypointCoordinates[0], waypointCoordinates[1]])
            .setPopup(new mapboxgl.Popup().setText(step.maneuver.instruction)) //Optional: Show instructions in a popup
            .addTo(map);

        //Store the marker in the waypointMarkers array for later removal
        waypointMarkers.push(waypointMarker);
    });

    //Log the complete waypoints array to the console
    console.log('Waypoints being sent:', waypoints);

    return waypoints
}

function extractMyWaypoints(myData){
    //Clear the previous waypoints array
    myWaypoints = [];

    //Remove previous markers from the map
    myWaypointMarkers.forEach(marker => marker.remove());
    myWaypointMarkers = []; //Clear the array of markers

    const steps = myData.legs[0].steps;

    //Loop through each step to extract waypoints
    steps.forEach(step => {
        const waypointCoordinates = step.maneuver.location; //[longitude, latitude]

        //Add the waypoint to the array
        myWaypoints.push(waypointCoordinates);

        const marker = document.createElement('div');
        marker.className = 'waypoint-marker-current'; //Add a class for styling

        //Create the marker and set its location
        const waypointMarker = new mapboxgl.Marker(marker)
            .setLngLat([waypointCoordinates[0], waypointCoordinates[1]])
            .setPopup(new mapboxgl.Popup().setText(step.maneuver.instruction)) //Optional: Show instructions in a popup
            .addTo(map);

        //Store the marker in the waypointMarkers array for later removal
        myWaypointMarkers.push(waypointMarker);
    });

    //Log the complete waypoints array to the console
    console.log('Waypoints being sent:', myWaypoints);

    return waypoints
}

function sendWaypointsToServer(waypoints) {
    fetch('/update_waypoints', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            waypoints: waypoints //Sending the array of waypoint coordinates
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log(waypoints)
        console.log('Coordinates successfully sent to server:', data);
    })
    .catch((error) => {
        console.error('Error sending coordinates to server:', error);
    });
}

function calculateTurningAngle(data){
    const steps = data.legs[0].steps;
    turnAngle = [];
    let angle;

    for(let i = 0; i < steps.length; i++){
        let bearingBefore = steps[i].maneuver.bearing_before;
        let bearingAfter = steps[i].maneuver.bearing_after;
        let location = steps[i].maneuver.location;
        let type = steps[i].maneuver.type;
        let turnOri;

        if (type == 'turn'){

            angle = bearingAfter - bearingBefore;

            if (angle < 0){ //Left turn - it was measure counterclockwise so left turn could be huge numbers(270) when the turn only needs to be 90
                angle = 180 - Math.abs(angle);
                turnOri = 'left';
            }
            else {
                turnOri = 'right';
            }
    
            turnAngle.push({
                angle: angle,
                location: location,
                type: type,
                turnOri: turnOri
            });
        }
    }

    console.log('Turn Data:', turnAngle);
    return turnAngle
}

function sendTurnAngleToServer(turnAngle){
    fetch('/update_turnAngle', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            turnAngle: turnAngle //Sending the array of waypoint coordinates
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('turnAngle successfully sent to server:', data);
    })
    .catch((error) => {
        console.error('Error sending turnAngle to server:', error);
    });
}

function routeBearing(myRoute) {
    bearingWLocations = [];

    if (myRoute.length < 2) {
        console.log("Insufficient points to calculate bearing.");
        return bearingWLocations;
    }

    //Initial bearings
    let lat1 = myRoute[0][1];
    let lon1 = myRoute[0][0];
    let lat2 = myRoute[1][1];
    let lon2 = myRoute[1][0];

    let startLoc = myRoute[0];
    let prevBearing = haversineBearing(lat1, lon1, lat2, lon2).bearing; //Extract bearing

    //console.log("Initial Bearing:", prevBearing, " at location:", startLoc);  //Debugging line

    for (let i = 1; i < myRoute.length - 1; i++) {
        lat1 = myRoute[i][1];
        lon1 = myRoute[i][0];
        lat2 = myRoute[i + 1][1];
        lon2 = myRoute[i + 1][0];

        let currBearing = haversineBearing(lat1, lon1, lat2, lon2).bearing; //Extract bearing

        //Log current bearings and difference
        //console.log(`Prev: ${prevBearing}, Curr: ${currBearing}, Difference: ${Math.abs(currBearing - prevBearing)}`);

        //Normalize bearing difference (handle cases where it crosses 360 degrees)
        let bearingDifference = Math.abs(currBearing - prevBearing);
        if (bearingDifference > 180) {
            bearingDifference = 360 - bearingDifference;
        }

        //Check for a significant change in direction
        if (bearingDifference > 15) {  //15-degree threshold
            console.log("Significant change detected at location:", startLoc);
            bearingWLocations.push({
                bearing: prevBearing,
                location: startLoc
            });

            //Update the start location and bearing
            startLoc = myRoute[i];
            prevBearing = currBearing;
        }
    }

    //Append the final bearing and location
    bearingWLocations.push({
        bearing: prevBearing,
        location: startLoc
    });

    console.log('Bearing with Locations: ', bearingWLocations);
    return bearingWLocations;
}

function sendRouteBearingToServer(bearingWLocations){
    fetch('/update_bearingWLocations', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            bearingWLocations: bearingWLocations //Sending the array of waypoint coordinates
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('bearingWLocations successfully sent to server:', data);
    })
    .catch((error) => {
        console.error('Error sending bearingWLocations to server:', error);
    });
}

//Function to calculate bearing and distance between two points on Earth
function haversineBearing(lat1, lon1, lat2, lon2) {
    //Convert degrees to radians
    lat1 = toRadians(lat1);
    lon1 = toRadians(lon1);
    lat2 = toRadians(lat2);
    lon2 = toRadians(lon2);

    const dlat = lat2 - lat1;
    const dlon = lon2 - lon1;

    const x = Math.sin(dlon) * Math.cos(lat2);
    const y = Math.cos(lat1) * Math.sin(lat2) - (Math.sin(lat1) * Math.cos(lat2) * Math.cos(dlon));
    const initialBearing = Math.atan2(x, y);
    const compassBearing = (toDegrees(initialBearing) + 360) % 360;

    const R = 6371; //Radius of Earth in kilometers
    const a = (Math.sin(dlat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dlon / 2) ** 2);
    const distance = R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)) * 1000; //Distance in meters

    return {
        distance: distance, //Distance in meters
        bearing: compassBearing //Bearing in degrees
    };
}

//Helper function to convert degrees to radians
function toRadians(degrees) {
    return degrees * (Math.PI / 180);
}

//Helper function to convert radians to degrees
function toDegrees(radians) {
    return radians * (180 / Math.PI);
}

//Function to fetch currentStep from the server
async function fetchCurrentStep() {
    try {
        const response = await fetch('/currentStep');  //GET request to your Flask route
        if (!response.ok) {
            const errorData = await response.json();
            console.error('Error fetching currentStep:', errorData.message);
            return;
        }

        const data = await response.json();
        console.log('currentStep received:', data);
        
        //Assuming you want to use the received coordinates
        currentStep = data
        console.log(currentStep)
        
    } 
    catch (error) {
        console.error('Fetch error:', error);
    }
}

function showCurrentInstruction(instructions, currentStep, waypoints, currentLat, currentLon) {
    console.log(currentStep);
    const instructionsListMid = document.getElementById('instructions-list-mid'); // Element to show shuffled instructions
    instructionsListMid.innerHTML = ''; // Clear any previous instructions
    
    const turnLon = waypoints[currentStep][0];
    const turnLat = waypoints[currentStep][1];

    console.log(`turnLon: ${turnLon}, turnLat: ${turnLat}`);

    const distanceToTurn = haversineBearing(currentLat, currentLon, turnLat, turnLon).distance;
    console.log(distanceToTurn);

    if (currentStep >= 0 && currentStep < instructions.length) {
        let instructionText = instructions[currentStep].replace(/\.$/,'');
        instructionsListMid.textContent = `${instructionText} for ${distanceToTurn}m`;
    } else {
        // Fallback if no instruction is available
        const li = document.createElement('li');
        li.textContent = `Keep going straight for ${distanceToTurn}`;
        instructionsListMid.appendChild(li);
    }
    
    console.log(instructionsListMid);
    fetchCurrentStep();
}

//Handle elements on the map
map.on('load', () => {
    //map.setBearing(15);//Set the desired angle in degrees
    map.addImage('pulsing-dot', pulsingDot, { pixelRatio: 2 });

    map.addSource('dot-point', {
        'type': 'geojson',
        'data': {
            'type': 'FeatureCollection',
            'features': [
                {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': current //icon position [lng, lat]
                    }
                }
            ]
        }
    });
    map.addLayer({
        'id': 'layer-with-pulsing-dot',
        'type': 'symbol',
        'source': 'dot-point',
        'layout': {
            'icon-image': 'pulsing-dot'
        }
    });

    setInterval(fetchCoordinates, 500); //Fetch every 0.5 seconds
    if(routeActive){
        setInterval(showCurrentInstruction, 500);
    }
});

//Handle clicks on the map
map.on('click', (event) => {
    const coords = [parseFloat(event.lngLat.lng), parseFloat(event.lngLat.lat)]; //Ensure coordinates are floats

    //Check which search box is active and update accordingly
    if (isDestinationSearchBox) {
        end = coords; //Update 'end' with clicked coordinates
        reverseGeocode(coords); //Reverse geocode to get the address
        updateMarkers(); //Update markers for current location
    } else if (isCurrentSearchBox) {
        start = coords; //Update 'start' with clicked coordinates
        reverseGeocode(coords); //Reverse geocode to get the address
        updateMarkers(); //Update markers for current location
    }
});

//Call the fetchCoordinates function on window load or at a relevant time
window.onload = fetchCoordinates();  //This will run the function when the page loads
