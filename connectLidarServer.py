import socket

def start_server():
    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to localhost and port 8080
    server_address = ('localhost', 8080)
    server_socket.bind(server_address)

    # Listen for incoming connections
    server_socket.listen(1)
    print("Server is waiting for a connection...")

    # Accept a connection
    connection, client_address = server_socket.accept()

    try:
        print("Connected to:", client_address)
        while True:
            # Receive data from the client
            data = connection.recv(1024)
            if data:
                print(f"Received: {data.decode()}")
            else:
                break
    finally:
        # Clean up the connection
        connection.close()

if __name__ == "__main__":
    start_server()
