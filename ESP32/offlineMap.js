let startMarker = null;
let destinationMarker = null;
let start = [];
let startLat = null;
let startLon = null;
let end = [];
let endLat = null;
let endLon = null;

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

//====================== Search, Start point, and Destination search boxes ======================== //
const searchBoxContainer = document.getElementById('searchBoxContainer');
let isSingleQueryMode = true; // Tracks which mode we're in

// Attach listeners for the Normal Search box
const attachQueryBoxListeners = () => {
    const queryBox = document.getElementById('queryBox');
    queryBox.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            console.log(`Query = ${queryBox.value}`);
            // Implement Nominatim query logic here
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

            if (startMarker) startMarker.remove();
            startMarker = new maplibregl.Marker({ color: '#28A228' })
                .setLngLat(start)
                .addTo(map);
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

            if (destinationMarker) destinationMarker.remove();
            destinationMarker = new maplibregl.Marker({ color: '#FF6961' })
                .setLngLat(end)
                .addTo(map);
        }
    });
};


// Toggle between Normal Search and Start/Destination modes
const toggleSearchBoxes = () => {
    const filterButtons = document.querySelector('.button-container');
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
        filterButtons.style.display = 'flex';
        routingButtons.style.display = 'flex';
    } else {
        // Switch back to single query mode
        searchBoxContainer.innerHTML = `
            <input id="queryBox" class="search-box" type="text" placeholder="Enter your query" />
        `;
        attachQueryBoxListeners();

        // Hide filter buttons
        filterButtons.style.display = 'none';
        routingButtons.style.display = 'none';
    }

    // Toggle the mode
    isSingleQueryMode = !isSingleQueryMode;
};

// Initial setup
document.getElementById('toggleSearchBoxes').addEventListener('click', toggleSearchBoxes);
attachQueryBoxListeners();
//=============================================================================================== //

//================================== FILTER ROUTE TYPE BUTTONS ================================== //
document.querySelectorAll('.btn').forEach(button => {
    button.addEventListener('click', () => {
        // Remove 'active' class from all buttons
        document.querySelectorAll('.btn').forEach(btn => btn.classList.remove('active'));
        if (button.textContent.trim() == 'Walking') {
            routeChoice = 'foot'
        }
        else if (button.textContent.trim() == 'Bicycle'){
            routeChoice = 'bicycle'
        }
        else {
            routeChoice = 'car'
        }

        console.log(routeChoice);

        // Add 'active' class to the clicked button
        button.classList.add('active');
    });
});
//=============================================================================================== //

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
// Add route to map
function addRouteToMap(route) {
    const geojson = {
        type: "Feature",
        properties: {},
        geometry: route.geometry, // Use geometry from the OSRM response
    };

    // Check if the route layer already exists
    if (map.getSource("route")) {
        map.getSource("route").setData(geojson); // Update the route
    } else {
        // Add route source
        map.addSource("route", {
            type: "geojson",
            data: geojson,
        });

        // Add route layer
        map.addLayer({
            id: "route",
            type: "line",
            source: "route",
            layout: {
                "line-cap": "round",
                "line-join": "round",
            },
            paint: {
                "line-color": "#0074D9", // Adjust route color
                "line-width": 5,        // Adjust route width
            },
        });
    }
}

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
};

//=====================================  DIRECTIONS BUTTON =======================================//
const viewRoutesButton = document.getElementById('view-route');
if (viewRoutesButton) {
    viewRoutesButton.addEventListener('click', async () => {

        isStartSearchBox = false;
        isEndSearchBox = false;

        // if (start.length === 0) {
        //     start = [...current];
        //     midRoute = false
        // } else if (end.length === 0) {
        //     end = [...current];
        //     midRoute = false
        // }
        // else {
        //     midRoute = true
        // }

        const [startLon, startLat] = start;
        const [endLon, endLat] = end;

        console.log(`Start route from (${startLat}, ${startLon}) to (${endLat}, ${endLon})`);
        
        viewRoutes = true;

        const dataToSend = {
            startLat: startLat,
            startLon: startLon,
            endLat: endLat,
            endLon: endLon
        };

        //Send data to websocket
        // socket.send(JSON.stringify(dataToSend));
        // console.log("Sent start and end coordinates to server:", dataToSend);

        updateMarkers();
        route = await routeOSRM(start, end, routeChoice); //Route full route
        console.log(`Midroute: ${midRoute}`);
        addRouteToMap(route);
        console.log(route);
        // if (midRoute){
        //     routeOSRM();    //Route user route too
        // }
    });
} 
else {
    console.error('View Routes button not found in the DOM');
}
//===============================================================================================//
