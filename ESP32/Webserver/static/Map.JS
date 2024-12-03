let startMarker = null;
let destinationMarker = null;
let queryMarker = null;

let start = [];
let startLat = null;
let startLon = null;
let end = [];
let endLat = null;
let endLon = null;
let query = [];
let current = []

let isStartSearchBox = false;
let isEndSearchBox = false;
let isQuerySearchBox = false;

let route = [];
let searchBox = null;

let directions = [];
let waypoints = [];
let waypointMarkersList = [];

let usrDirections = [];
let usrWaypoints = [];
let usrWaypointMarkersList = [];
let usrModifier = [];
let usrBearingBefore = [];
let usrBearingAfter = [];
let usrTurnTypes = [];

let midRoute = false;
let currentTab = 'fullRoute'; // Default to 'fullRoute'

let currentStep = 0;
let rotation = 0;

let currentData = null;
let msg;
let distanceToTurn;

const socket = new WebSocket('ws://127.0.0.1:5501/ws');

// Initialize MapLibre map
const map = new maplibregl.Map({
    container: 'map', // HTML container ID
    style: 'http://localhost:3650/api/maps/streets/style.json', // MapTiler Server style URL
    center: [103.82, 1.35], // Initial map center [lng, lat]
    zoom: 11.5, // Initial zoom level
});

// Set the initial rotations based on the initial map bearing
const initialRotation = map.getBearing();
console.log(initialRotation)
document.querySelector('#cardinal-directions').style.transform = `rotate(${initialRotation}deg)`;

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
        
        console.log(`Map recentred to current location: (${currentLat}, ${currentLon})`);
    } 
    else {
        console.error('Current location not available to recentre the map.');
    }
}
//==================================== PURPLE CURRENT LOCATION ====================================//
let currentLat = current[1];
let currentLon = current[0];

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

//Compass directions
function updateCompassDirection(azimuth, rotation){
    const updatedAzimuth = azimuth + rotation;
    if (updatedAzimuth > 360) {
        azimuth = updatedAzimuth - 360;
    }

    return azimuth;
}

//Orientate map
map.on('rotate', function() {
    rotation = map.getBearing(); // Get current map bearing (rotation)

    // Rotate the cardinal directions around their center
    const cardinalDirections = document.querySelector('#cardinal-directions');
    cardinalDirections.style.transform = `rotate(${rotation}deg)`;
});

async function updateDotWithCoordinates(data) {
    try {
        // Ensure data contains the necessary properties
        const newCoordinates = [data.longitude, data.latitude];
        console.log("new coordinates: ", newCoordinates);
        
        // Only perform the animation if coordinates have changed
        if (!areCoordinatesEqual(current, newCoordinates)) {
            // Start smooth transition to the new coordinates
            smoothUpdateDot(current, newCoordinates);
            // Update the current position
            current = newCoordinates.slice();
            currentLat = data.currentLat;
            currentLon = data.currentLon;
            azimuth = data.azimuth; // Update azimuth if needed
            //usrDirection = data.direction; // Update user direction if needed
            
            console.log('Current location:', current);
            console.log('Angle:', azimuth);
            //console.log('userDirection:', usrDirection);
        }

        // Update the triangle's orientation
        const triangle = document.getElementById('triangle');
        if (triangle) {
            azimuth = 20;
            let updatedAzimuth = azimuth + rotation;
            if (updatedAzimuth > 360) {
                updatedAzimuth -= 360;
            }

            updateTrianglePosition(current, updatedAzimuth);  // Update location and angle
        } 
        else {
            console.error('Triangle element not found');
        }
    } catch (error) {
        console.error('Fetch error:', error);
    }
}

// Smoothly update the dot's position on the map
function smoothUpdateDot(startCoords, endCoords) {
    const duration = 1000; // Duration of the transition in milliseconds
    const startTime = performance.now(); // Start time for the animation

    function animate(timestamp) {
        const progress = Math.min((timestamp - startTime) / duration, 1); // Calculate progress

        // Interpolate longitude and latitude
        const lng = startCoords[0] + (endCoords[0] - startCoords[0]) * progress;
        const lat = startCoords[1] + (endCoords[1] - startCoords[1]) * progress;

        // Update the GeoJSON data for the dot
        map.getSource('dot-point').setData({
            'type': 'FeatureCollection',
            'features': [
                {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [lng, lat] // Update coordinates with interpolated values
                    }
                }
            ]
        });

        if (progress < 1) {
            requestAnimationFrame(animate); // Continue the animation until it completes
        } 
        else {
            // When the animation completes, set the current to the end coordinates
            current = endCoords.slice();
        }
    }

    requestAnimationFrame(animate); // Start the animation
}

// Helper function to compare coordinates
function areCoordinatesEqual(coord1, coord2) {
    return coord1[0] === coord2[0] && coord1[1] === coord2[1];
}
//================================================================================================//

//====================== SEARCH, START POINT AND DESTINATION SEARCH BOXES ========================//
const searchBoxContainer = document.getElementById('searchBoxContainer');
let isSingleQueryMode = true; // Tracks which mode we're in

//Attach listeners for the Normal Search box
const attachQueryBoxListeners = () => {
    const queryBox = document.getElementById('queryBox');

    // Focus listener for the start box
    queryBox.addEventListener('focus', () => {
        isQuerySearchBox = true;
        isStartSearchBox = false;
        isEndSearchBox = false;
    });

    queryBox.addEventListener('keydown', async (event) => {
        if (event.key === 'Enter') {
            console.log(`Query = ${queryBox.value}`);
            // Implement Nominatim query logic here
            const data = await geocodeNominatim(queryBox.value, 'forward', 'query');
            console.log(data);

            query = [parseFloat(data.lon), parseFloat(data.lat)];

            map.flyTo({
                center: query,
                zoom: 11.5,
                speed: 1.2,
                curve: 1,
            });

            isStartSearchBox = false;
            isEndSearchBox = false;
            isQuerySearchBox = false;

            updateMarkers();
        }
    });
};

//Attach listeners for Start and Destination boxes
const attachStartAndDestinationListeners = () => {
    const startBox = document.getElementById('startBox');
    const destinationBox = document.getElementById('destinationBox');

    // Focus listener for the start box
    startBox.addEventListener('focus', () => {
        isStartSearchBox = true;
        isEndSearchBox = false;
    });

    // Escape key listener for the start box
    startBox.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            isStartSearchBox = false;
            isEndSearchBox = false;
        }
    });

    // Enter key listener for the start box
    startBox.addEventListener('keydown', async (event) => {
        if (event.key === 'Enter') {
            console.log(`Start point = ${startBox.value}`);
            const data = await geocodeNominatim(startBox.value, 'forward', 'start');
            console.log(data);

            startLat = parseFloat(data.lat);
            startLon = parseFloat(data.lon);
            start = [startLon, startLat];

            map.flyTo({
                center: start,
                zoom: 9,
                speed: 1.2,
                curve: 1,
            });

            isStartSearchBox = false;
            isEndSearchBox = false;

            updateMarkers();

            await triggerRouting();
        }
    });

    // Focus listener for the destination box
    destinationBox.addEventListener('focus', () => {
        isStartSearchBox = false;
        isEndSearchBox = true;
    });

    // Escape key listener for the destination box
    destinationBox.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            isStartSearchBox = false;
            isEndSearchBox = false;
        }
    });

    // Enter key listener for the destination box
    destinationBox.addEventListener('keydown', async (event) => {
        if (event.key === 'Enter') {
            console.log(`Destination = ${destinationBox.value}`);
            const data = await geocodeNominatim(destinationBox.value, 'forward', 'end');

            endLat = parseFloat(data.lat);
            endLon = parseFloat(data.lon);
            end = [endLon, endLat];

            map.flyTo({
                center: end,
                zoom: 9,
                speed: 1.2,
                curve: 1,
            });

            updateMarkers();

            isStartSearchBox = false;
            isEndSearchBox = false;
            
            await triggerRouting();
        }
    });
};

//Toggle between Normal Search and Start/Destination modes
const toggleSearchBoxes = () => {
    const routingButtons = document.querySelector('.routing-container');

    if (isSingleQueryMode) {
        // Switch to Start/Destination boxes mode
        searchBoxContainer.innerHTML = `
            <div class="start-destination-container">
                <div>
                    <input id="startBox" class="search-box" type="text" placeholder="Enter start point" />
                </div>
                <div>
                    <input id="destinationBox" class="search-box" type="text" placeholder="Enter destination" />
                </div>
            </div>
        `;
        // Call the listeners after new elements are added
        attachStartAndDestinationListeners();

        // Hide the query marker if it exists
        if (queryMarker) {
            queryMarker.remove();
            queryMarker = null; // Ensure it's cleared to avoid reuse
        }

        // Show filter buttons
        routingButtons.style.display = 'flex';
        document.getElementById('clear-markers').addEventListener('click', clearAllMarkersAndRoutes);
    } 
    else {
        // Switch back to single query mode
        searchBoxContainer.innerHTML = `
            <input id="queryBox" class="search-box" type="text" placeholder="Enter your query" />
        `;
        attachQueryBoxListeners();

        // Restore the query marker if a query exists
        if (query.length) {
            queryMarker = new maplibregl.Marker({ color: '#7F8389' })
                .setLngLat(query)
                .addTo(map);
        }

        // Hide filter buttons
        routingButtons.style.display = 'none';
    }

    // Toggle the mode
    isSingleQueryMode = !isSingleQueryMode;

    updateMarkers();
};

// Initial setup
document.getElementById('toggleSearchBoxes').addEventListener('click', toggleSearchBoxes);
attachQueryBoxListeners();
//===============================================================================================//


//========================================== GEOCODING ===========================================//
//Function to fetch geocode results from Nominatim
async function geocodeNominatim(query, type, searchBox) {
    let url;
    let coords;
    console.log(searchBox);

    if (type == 'forward'){
        url= `http://localhost:8088/search.php?q=${encodeURIComponent(query)}&format=json&addressdetails=1`;
        console.log(url)
    }
    else if (type == 'reverse'){
        coords = query;
        console.log(coords)
        url = `http://localhost:8088/reverse.php?lat=${coords[1]}&lon=${coords[0]}&format=json&addressdetails=1`;

        if (searchBox == 'start'){
            start = coords;
            console.log('start')
        }
        else if (searchBox == 'end'){
            end = coords;
            console.log('end')
        }
        else if (searchBox == 'query'){
            query = coords;
        }
        await triggerRouting();
    }

    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
            const data = await response.json();
        if (data.length === 0) throw new Error('No results found');
        if (type == 'forward'){
            console.log(data[0])
            return data[0]; // Return the first result
        }
        else if (type == 'reverse'){
            return data; // Return the first result
        }
    } catch (error) {
        console.error('Geocoding error:', error);
        throw error;
    }

    console.log("trigger routing");

    await triggerRouting();
}
//================================================================================================//


//=========================================== ROUTING ============================================//
const triggerRouting = async () =>{
    if (start.length && end.length){
        console.log(`Route from (${start[1]}, ${start[0]}) to (${end[1]}, ${end[0]})`);
        midRoute = true;
    }
    else if (start.length || end.length){ //If only start is filled
        midRoute = false;
    }
    else {
        console.error('No valid start or end points available')
    }

    try{
        usrRoute = await routeOSRM(current, end);   //Always route usr route
        addRouteToMap(usrRoute, "#A020F0", "usrRoute");
        console.log('User: ', usrRoute);
        extractAttributes(usrRoute, "usrRoute")

        console.log(midRoute)
        if (midRoute){
            route = await routeOSRM(start, end);    //If current is not start pnt, route full route too
            console.log('Route: ', route);
            addRouteToMap(route, "#0074D9", "fullRoute");
            extractAttributes(route, "fullRoute")
        }

        // Zoom to fit route bounds
        const bounds = getRouteBounds(usrRoute.geometry.coordinates);
        map.fitBounds(bounds, {
            padding: 10, // Padding around the route
            maxZoom: 15, // Limit the zoom level
            duration: 1000, // Animation duration in milliseconds
        });
    }
    catch (error) {
        console.error('Error fetching route:', error);
    }
}

//Function to fetch route results from OSRM
async function routeOSRM(start, end) {
    const url = `http://localhost:5000/route/v1/car/${start[0]},${start[1]};${end[0]},${end[1]}?overview=full&geometries=geojson&steps=true`;
    console.log(url)
    console.log(start)
    console.log(end)
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`OSRM Error: ${response.statusText}`);
            const data = await response.json();
        if (!data.routes || data.routes.length === 0) throw new Error('No route found');
            return data.routes[0]; // Return the first route
    } catch (error) {
        console.error('Routing error:', error);
        throw error;
    }
}

//Add route to map
function addRouteToMap(route, colour, routeID) {
    const geojson = {
        type: "Feature",
        properties: {},
        geometry: route.geometry, // Use geometry from the OSRM response
    };

    // Check if the route layer already exists
    if (map.getSource(routeID)) {
        map.getSource(routeID).setData(geojson); // Update the route
    } else {
        // Add route source
        map.addSource(routeID, {
            type: "geojson",
            data: geojson,
        });

        // Add route layer
        map.addLayer({
            id: routeID,
            type: "line",
            source: routeID,
            layout: {
                "line-cap": "round",
                "line-join": "round",
            },
            paint: {
                "line-color": colour, // Adjust route colour
                "line-width": 5,        // Adjust route width
            },
        });
    }
}

const getRouteBounds = (coordinates) => {
    const bounds = coordinates.reduce(
        (acc, [lon, lat]) => {
            acc[0][0] = Math.min(acc[0][0], lon); // Westmost longitude
            acc[0][1] = Math.min(acc[0][1], lat); // Southmost latitude
            acc[1][0] = Math.max(acc[1][0], lon); // Eastmost longitude
            acc[1][1] = Math.max(acc[1][1], lat); // Northmost latitude
            return acc;
        },
        [[Infinity, Infinity], [-Infinity, -Infinity]]
    );
    return bounds;
};
//================================================================================================//


//================================= DISPLAY INSTRUCTIONS BUTTON =================================//
document.addEventListener("DOMContentLoaded", () => {
    //Get the toggle button and the instructions container
    const toggleButton = document.getElementById('display-instr');
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


//============================== TOGGLE INSTRUCTIONS STYLE BUTTON ==============================//
document.getElementById('toggle-instr').addEventListener('click', function () {
    this.classList.toggle('active');
    const shuffleMode = this.classList.contains('active');

    const instructionsListFull = document.getElementById('instructions-list-full');
    const instructionsListMid = document.getElementById('instructions-list-mid');
    const hiddenCurrentInstruction = document.getElementById('hidden-current-instruction');

    if (shuffleMode) {
        // Shuffle mode ON: Hide instruction lists and show current instruction
        instructionsListFull.style.display = 'none';
        instructionsListMid.style.display = 'none';
        hiddenCurrentInstruction.style.display = 'block';

        // Show current instruction (assumes `currentStep`, `usrDirections`, and `usrWaypoints` are defined globally)
        showCurrentInstruction(usrDirections, usrWaypoints, msg, distanceToTurn, currentStep);

        console.log('Shuffle mode ON: Showing current instruction.');
    } else {
        // Shuffle mode OFF: Restore instruction lists and hide current instruction
        instructionsListFull.style.display = 'block';
        instructionsListMid.style.display = 'block';
        hiddenCurrentInstruction.style.display = 'none';

        console.log('Shuffle mode OFF: Restoring instruction lists.');
    }
});

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
        } 
        else {
            // Set styles for the inactive tab button
            tabButton.classList.remove('tab-button-active');
            tabButton.classList.add('tab-button-inactive');
        }
    });

    console.log('Current Tab:', currentTab); // This will print the current active tab name
}
//===============================================================================================//

//==================================== CLEAR MARKERS BUTTON =====================================//
const clearAllMarkersAndRoutes = () => {
    // Remove start marker
    if (startMarker) {
        startMarker.remove();
        startMarker = null;
    }

    // Remove destination marker
    if (destinationMarker) {
        destinationMarker.remove();
        destinationMarker = null;
    }

    // Remove query marker
    if (queryMarker) {
        queryMarker.remove();
        queryMarker = null;
    }

    const destinationBox = document.getElementById('destinationBox');
    const startBox = document.getElementById('startBox');
    destinationBox.value = '';
    startBox.value = '';

    // Remove waypoint markers
    waypointMarkersList.forEach(marker => marker.remove());
    waypointMarkersList = [];
    usrWaypointMarkersList.forEach(marker => marker.remove());
    usrWaypointMarkersList = [];

    // Remove route layer and source
    if (map.getLayer("fullRoute")) {
        map.removeLayer("fullRoute");
    }
    if (map.getLayer("usrRoute")) {
        map.removeLayer("usrRoute");
    }
    if (map.getSource("fullRoute")) {
        map.removeSource("fullRoute");
    }
    if (map.getSource("usrRoute")) {
        map.removeSource("usrRoute");
    }

    // Clear associated variables
    start = [];
    end = [];
    query = [];
    route = [];
    waypoints = [];
    usrWaypoints = [];
    directions = [];
    usrDirections = [];

    console.log("All markers and routes cleared.");
};
//===============================================================================================//

//====================================== START ROUTE BUTTON =====================================//
let routeActive = false;
const startRouteButton = document.getElementById('start-route');
const clearMarkersButton = document.getElementById('clear-markers');
const toggleInstrButton = document.getElementById('toggle-instr');
toggleInstrButton.disabled = true;

if (startRouteButton && clearMarkersButton) {
    startRouteButton.addEventListener('click', () => {
        // Locate the span that contains the button text
        const textSpan = startRouteButton.querySelector('span:nth-of-type(2)');

        if (!routeActive) {
            // Change button text to 'Stop'
            textSpan.innerText = "Stop";
            routeActive = true; // Route has started
            
            // Enable the shuffle instructions style button when route is active
            toggleInstrButton.disabled = false;
            // Disable the clear markers button when the route is active
            clearMarkersButton.disabled = true;

            // Send all to server when starting the route
            sendToServer(usrWaypoints, usrTurnTypes, usrModifier, usrBearingAfter, usrBearingBefore);

        } else {
            // Change button text to 'Start'
            textSpan.innerText = "Start";
            routeActive = false;

            clearMarkersButton.disabled = false;
            toggleInstrButton.disabled = true;

            start = [];
            end = [];
        }
        sendRouteActive(); // Notify the server of route status
    });
} else {
    console.error('Start Route button not found in the DOM');
}

function sendRouteActive(){
    const message = JSON.stringify({
        type: 'routeActive',
        routeActive: routeActive // Send the array of instructions
    });

    socket.send(message)
    console.log('routeActive sent to server:', routeActive);
}
//===============================================================================================//
function extractAttributes(data, routeType, shuffleMode = false) {
    const instructionsListFull = document.getElementById('instructions-list-full');
    const instructionsListMid = document.getElementById('instructions-list-mid');
    const hiddenCurrentInstruction = document.getElementById('hidden-current-instruction');

    const steps = data.legs[0].steps;

    // Clear instructions and reset markers/directions based on routeType
    if (routeType === "usrRoute") {
        instructionsListMid.innerHTML = '';
        usrDirections = [];
        usrWaypointMarkersList.forEach(marker => marker.remove());
        usrWaypointMarkersList = [];
    } else if (routeType === "fullRoute") {
        instructionsListFull.innerHTML = '';
        directions = [];
        waypointMarkersList.forEach(marker => marker.remove());
        waypointMarkersList = [];
    }

    // Process steps for the route
    for (const step of steps) {
        const { name, maneuver } = step;
        const { type, modifier, location, bearing_before, bearing_after } = maneuver;

        // Add waypoints based on routeType
        if (routeType === "usrRoute") {
            usrWaypoints.push(location);
            console.log(usrWaypoints)
            usrTurnTypes.push(type)
            usrModifier.push(modifier);
            usrBearingBefore.push(bearing_before)
            usrBearingAfter.push(bearing_after)
        } else if (routeType === "fullRoute") {
            waypoints.push(location);
        }

        // Construct a readable instruction
        let instruction = `${type.charAt(0).toUpperCase() + type.slice(1)} `;
        if (modifier) {
            instruction += `${modifier} `;
        }
        instruction += name ? `onto ${name}.` : "ahead.";

        if (type === "depart") {
            instruction = name ? `Start from ${name}.` : "Start.";
        }

        // Add instructions and markers based on routeType
        if (routeType === "usrRoute") {
            const liMid = document.createElement('li');
            liMid.textContent = instruction;
            instructionsListMid.appendChild(liMid);

            usrDirections.push(instruction);

            const usrMarkerElement = createMarkerElement('rgba(174, 48, 240, 1)');
            const usrWaypointMarker = new maplibregl.Marker({ element: usrMarkerElement })
                .setLngLat([location[0], location[1]])
                .addTo(map);
            usrWaypointMarkersList.push(usrWaypointMarker);
        } 
        else if (routeType === "fullRoute") {
            const liFull = document.createElement('li');
            liFull.textContent = instruction;
            instructionsListFull.appendChild(liFull);

            directions.push(instruction);

            const fullMarkerElement = createMarkerElement('gray');
            const waypointMarker = new maplibregl.Marker({ element: fullMarkerElement })
                .setLngLat([location[0], location[1]])
                .addTo(map);
            waypointMarkersList.push(waypointMarker);
        }
    }

    // Handle visibility based on shuffleMode
    if (shuffleMode) {
        instructionsListFull.style.display = 'none';
        instructionsListMid.style.display = 'none';
        hiddenCurrentInstruction.style.display = 'block';
    } else {
        instructionsListFull.style.display = 'block';
        instructionsListMid.style.display = 'block';
        hiddenCurrentInstruction.style.display = 'none';
    }
}

// Helper function to create marker elements
function createMarkerElement(colour) {
    const markerElement = document.createElement('div');
    markerElement.style.width = '10px';
    markerElement.style.height = '10px';
    markerElement.style.backgroundColor = colour;
    markerElement.style.borderRadius = '50%';
    markerElement.style.border = '2px solid white';
    return markerElement;
}

//Function that outputs instructions based on current/testStep in console instead of on display for now
function showCurrentInstruction(instructions, waypoints, msg, distanceToTurn, currentStep) {
    console.log(instructions)
    console.log(waypoints)
    const totalSteps = instructions.length;
    // currentStep = currentStepObj.currentStep
    console.log('currentStep: ', currentStep);
    console.log('totalSteps: ', totalSteps)
    
    const hiddenCurrentInstruction = document.getElementById('hidden-current-instruction'); // Element to show shuffled instructions
    hiddenCurrentInstruction.innerHTML = ''; // Clear any previous instructions

    if (currentStep >= 0 && currentStep < totalSteps) {
        console.log('currentStep instructions')
        let turnStep = waypoints[currentStep]
        console.log(turnStep)

        const turnLon = turnStep[0];
        const turnLat = turnStep[1];

        console.log(`turnLon: ${turnLon}, turnLat: ${turnLat}`);

        console.log(distanceToTurn);

        if (distanceToTurn <= 1){
            let instructionText = instructions[currentStep].replace(/\.$/,'');
            hiddenCurrentInstruction.textContent = `${instructionText} in ${distanceToTurn}m`;
            console.log('currentStep: ', currentStep);
        }
        else{
            let instructionText = msg.replace(/\.$/,'');
            hiddenCurrentInstruction.textContent = `${instructionText}`;
            console.log('currentStep: ', currentStep);
        }

    } else {
        console.log('currentStep fallback')
        // Fallback if no instruction is available
        const li = document.createElement('li');
        li.textContent = `Keep going straight for ${distanceToTurn} meters`;
        //console.log(distanceToTurn.toFixed(1))
        hiddenCurrentInstruction.appendChild(li);
    }
    
    console.log('instructionsListMid: ', hiddenCurrentInstruction);
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
    const distance = R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)) * 100000; //Distance in meters

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

const updateMarkers = () => {
    //Remove existing markers if they exist
    if (startMarker) {
        startMarker.remove();
    }
    if (destinationMarker) {
        destinationMarker.remove();
    }
    if (queryMarker){
        queryMarker.remove();
    }

    //Create new start marker if a start location exists
    if (start.length) {
        startMarker = new maplibregl.Marker({ color: '#28A228' }) //Blue marker for start
            .setLngLat(start)
            .addTo(map);
    }

    //Create new destination marker if an end location exists
    if (end.length) {
        destinationMarker = new maplibregl.Marker({ color: '#FF6961' }) //Red marker for destination
            .setLngLat(end)
            .addTo(map);
    }

    //Create a new query marker if a query location exists
    if (isSingleQueryMode && query.length){
        queryMarker = new maplibregl.Marker({ color: '#7F8389' }) //Red marker for destination
            .setLngLat(query)
            .addTo(map);
    }
};

//Handle map loads
map.on('load', () => {
    console.log(current)
    //map.setBearing(15);//Set the desired angle in degrees
    map.addImage('pulsing-dot', pulsingDot, { pixelRatio: 2 });

    //console.log("purple dot")
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
});

//Handle clicks on the map
map.on('click', async (event) => {
    const coords = [parseFloat(event.lngLat.lng), parseFloat(event.lngLat.lat)]; // Ensure coordinates are floats

    if (isEndSearchBox) {
        end = coords; // Update 'end' with clicked coordinates
        try {
            const addressData = await geocodeNominatim(end, 'reverse', 'end'); // Reverse geocode to get the address
            console.log(addressData)
            const address = addressData.display_name; // Extract the display name (address)

            // Populate the destination box with the address
            const destinationBox = document.getElementById('destinationBox');
            if (destinationBox) {
                destinationBox.value = address;
            }

            updateMarkers(); // Update markers for current location
        } catch (error) {
            console.error('Error getting address for destination:', error);
        }
    } 
    else if (isStartSearchBox) {
        start = coords; // Update 'start' with clicked coordinates
        try {
            const addressData = await geocodeNominatim(start, 'reverse', 'start'); // Reverse geocode to get the address
            console.log(addressData)
            const address = addressData.display_name; // Extract the display name (address)

            // Populate the start box with the address
            const startBox = document.getElementById('startBox');
            if (startBox) {
                startBox.value = address;
            }

            updateMarkers(); // Update markers for current location
        } catch (error) {
            console.error('Error getting address for start:', error);
        }
    }

    else if (isQuerySearchBox) {
        query = coords;
        try {
            const addressData = await geocodeNominatim(query, 'reverse'); // Reverse geocode to get the address
            console.log(addressData)
            const address = addressData.display_name; // Extract the display name (address)

            // Populate the start box with the address
            const queryBox = document.getElementById('queryBox');
            if (queryBox) {
                queryBox.value = address;
            }

            updateMarkers(); // Update markers for current location
        } catch (error) {
            console.error('Error getting address for start:', error);
        }
    }
});

async function fetchCoordinates() {
    if (currentData) {
        console.log('Using current data:', currentData);
        // Update the UI or perform operations with currentData
        await updateDotWithCoordinates(currentData);
    }
}

async function sendToServer(usrWaypoints, usrTurnTypes, usrModifier, usrBearingAfter, usrBearingBefore){
    //Send one by one because of byte limit in sending to esp32
    //WAYPOINTS
    const waypointsPayload = JSON.stringify({
        type: 'w',
        waypoints: usrWaypoints,
    });

    socket.send(waypointsPayload)
    console.log('Attribute sent to server:', waypointsPayload);

    //MODIFIER
    const modifierPayload = JSON.stringify({
        type: 'modifier',
        modifier: usrModifier,
    });

    socket.send(modifierPayload)
    console.log('Attribute sent to server:', modifierPayload);

    //TURN TYPES
    const turnPayloads = JSON.stringify({
        type: 'turnTypes',
        turnTypes: usrTurnTypes,
    });

    socket.send(turnPayloads)
    console.log('Attribute sent to server:', turnPayloads);


    //BEARING BEFORE
    const bbPayload = JSON.stringify({
        type: 'bearingBefore',
        bearingBefore: usrBearingBefore,
    });

    socket.send(bbPayload)
    console.log('Attribute sent to server:', bbPayload);

    //BEARING AFTER
    const baPayload = JSON.stringify({
        type: 'bearingAfter',
        bearingAfter: usrBearingAfter,
    });

    socket.send(baPayload)
    console.log('Attribute sent to server:', baPayload);
}

//Initialise websockets
function socketListener(socket) {
    // Listen for incoming messages from the server
    socket.onmessage = (event) => {  // Changed from `socket.on` to `socket.onmessage`
        const data = JSON.parse(event.data);
        console.log('Data received: ', data);
        console.log('Data type: ', data.type);
        console.log('Data type2: ', data["type"]);
        console.log('Data type:', typeof data);

        if (data.type === "currentCoords") {
            console.log('curentCoords')
            const { type, ...rest } = data;
            currentData = { ...rest};
            console.log('Updated coordinates received:', currentData);
            fetchCoordinates();
        }

        else if (data.type === 'testStep') {
            const { type, ...rest } = data;
            currentStep = {...rest};
            console.log('Updated currentStep received:', currentStep);
            if (!midRoute){
                usrDirections = directions
                usrWaypoints = waypoints
            }
            showCurrentInstruction(usrDirections, usrWaypoints, currentLat, currentLon); // Call here so it's not dependednt on the button click
        }
        else if (data.type === 'vehicleControl') {
            const { type, ...rest } = data;
            instructionData = {...rest};
            console.log('Updated currentStep received:', instructionData);
            msg = instructionData["msg"];
            distanceToTurn = instructionData["distanceToTurn"];
            currentStep = instructionData["currentStep"];

            console.log("Distance to turn: ", distanceToTurn);

            showCurrentInstruction(usrDirections, usrWaypoints, msg, distanceToTurn, currentStep); // Call here so it's not dependednt on the button click
        }
        else {
            console.warn('Unknown data type received: ', data);
        }
    };
    
    socket.onclose = () => {
        console.log('WebSocket disconnected');
    };

    socket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// Call the socketListener function when the window loads
window.onload = function() {
    socketListener(socket);
    sendRouteActive();
}
