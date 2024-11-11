#boot.py
import network	#WiFi
import socket	#TCP communication
import time	#Delays

# WiFi and server settings
ssid = 'AndroidAP3a93'	#WiFi name
password = 'buhh1927'	#WiFi password

SERVER_IP = "192.168.230.96"  # Webserver IP
SERVER_PORT = 8765	# Webserver port, different from web application port

JETSON_IP = "192.168.167.167"	#Jetson IP
JETSON_PORT = 5000	#Jetson port

# Global client_socket variable
client_socket = None	#Webserver client socket
client_socket_jetson = None	#Jetson client socket

def connect_to_wifi():
    """Connect to the specified Wi-Fi network."""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(ssid, password)
    
    while not wlan.isconnected():
        print("Connecting to Wi-Fi...")
        time.sleep(1)
    
    print("Connected to Wi-Fi:", wlan.ifconfig())

def connect_to_socket():
    """Connect to the server."""
    global client_socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_IP, SERVER_PORT))
    print("Connected to server")
    
def connect_to_jetson():
    """Connect to the jeton."""
    global client_socket_jetson
    client_socket_jetson = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket_jetson.connect((JETSON_IP, JETSON_PORT))
    print("Connected to Jetson")

#connect_to_wifi()
#connect_to_socket()