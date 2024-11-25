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

let isStartSearchBox = false;
let isEndSearchBox = false;

let route = [];

let midRoute = false;

// Initialize MapLibre map
const map = new maplibregl.Map({
    container: 'map', // HTML container ID
    style: 'http://localhost:3650/api/maps/streets/style.json', // MapTiler Server style URL
    center: [103.82, 1.35], // Initial map center [lng, lat]
    zoom: 11.5, // Initial zoom level
});

//====================== Search, Start point, and Destination search boxes ========================//
const searchBoxContainer = document.getElementById('searchBoxContainer');
let isSingleQueryMode = true; // Tracks which mode we're in

// Attach listeners for the Normal Search box
const attachQueryBoxListeners = () => {
    const queryBox = document.getElementById('queryBox');
    queryBox.addEventListener('keydown', async (event) => {
        if (event.key === 'Enter') {
            console.log(`Query = ${queryBox.value}`);
            // Implement Nominatim query logic here
            const data = await geocodeNominatim(queryBox.value, 'forward');
            console.log(data);

            query = [parseFloat(data.lon), parseFloat(data.lat)];

            map.flyTo({
                center: query,
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
};

// Attach listeners for Start and Destination boxes
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
            const data = await geocodeNominatim(startBox.value, 'forward');
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
            const data = await geocodeNominatim(destinationBox.value, 'forward');

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

// Toggle between Normal Search and Start/Destination modes
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

        // Show filter buttons
        routingButtons.style.display = 'flex';
    } else {
        // Switch back to single query mode
        searchBoxContainer.innerHTML = `
            <input id="queryBox" class="search-box" type="text" placeholder="Enter your query" />
        `;
        attachQueryBoxListeners();

        // Hide filter buttons
        routingButtons.style.display = 'none';
    }

    // Toggle the mode
    isSingleQueryMode = !isSingleQueryMode;
};

// Initial setup
document.getElementById('toggleSearchBoxes').addEventListener('click', toggleSearchBoxes);
attachQueryBoxListeners();
//===============================================================================================//


//========================================== GEOCODING ==========================================//
//Function to fetch geocode results from Nominatim
async function geocodeNominatim(query, type) {
    let url;

    if (type == 'forward'){
        url= `http://localhost:8088/search.php?q=${encodeURIComponent(query)}&format=json&addressdetails=1`;
    }
    else if (type == 'reverse'){
        let coords = query;
        console.log(coords)
        url = `http://localhost:8088/reverse.php?lat=${coords[1]}&lon=${coords[0]}&format=json&addressdetails=1`;
    }

    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
            const data = await response.json();
        if (data.length === 0) throw new Error('No results found');
        if (type == 'forward'){
            return data[0]; // Return the first result
        }
        else if (type == 'reverse'){
            return data; // Return the first result
        }
    } catch (error) {
        console.error('Geocoding error:', error);
        throw error;
    }
}
//===============================================================================================//


//=========================================== ROUTING ===========================================//
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
        addRouteToMap(usrRoute, "A020F0", "usrRoute");
        console.log('Route: ', usrRoute);

        if (midRoute){
            route = await routeOSRM(start, end);    //If current is not start pnt, route full route too
            addRouteToMap(route, "#0074D9", "fullRoute");
        }
    }
    catch (error) {
        console.error('Error fetching route:', error);
    }
}


//Function to fetch route results from OSRM
async function routeOSRM(start, end, routeChoice) {
    const url = `http://localhost:5000/route/v1/${routeChoice}/${start[0]},${start[1]};${end[0]},${end[1]}?overview=full&geometries=geojson&steps=true`;
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
                "line-color": colour, // Adjust route color
                "line-width": 5,        // Adjust route width
            },
        });
    }
}
//===============================================================================================//

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


//============================== SHUFFLE INSTRUCTIONS STYLE BUTTON ==============================//
document.getElementById('toggle-instr').addEventListener('click', function() {
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

//Handle clicks on the map
map.on('click', async (event) => {
    const coords = [parseFloat(event.lngLat.lng), parseFloat(event.lngLat.lat)]; // Ensure coordinates are floats

    if (isEndSearchBox) {
        end = coords; // Update 'end' with clicked coordinates
        try {
            const addressData = await geocodeNominatim(end, 'reverse'); // Reverse geocode to get the address
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
    } else if (isStartSearchBox) {
        start = coords; // Update 'start' with clicked coordinates
        try {
            const addressData = await geocodeNominatim(start, 'reverse'); // Reverse geocode to get the address
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
});

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
    if (query.length){
        queryMarker = new maplibregl.Marker({ color: '#7F8389' }) //Red marker for destination
            .setLngLat(query)
            .addTo(map);
    }
};
