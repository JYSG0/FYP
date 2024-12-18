#include <iostream>
#include <winsock2.h>

#pragma comment(lib, "ws2_32.lib")  // Link the Winsock library

int main() {
    WSADATA wsaData;
    SOCKET sock;
    struct sockaddr_in server;

    // Initialize Winsock
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed with error: " << WSAGetLastError() << std::endl;
        return 1;
    }

    // Create a socket
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
        std::cerr << "Socket creation failed with error: " << WSAGetLastError() << std::endl;
        WSACleanup();
        return 1;
    }

    // Define the server address
    server.sin_family = AF_INET;
    server.sin_port = htons(8080);  // Port number
    server.sin_addr.s_addr = inet_addr("127.0.0.1");  // Localhost

    // Connect to the server
    if (connect(sock, (struct sockaddr *)&server, sizeof(server)) == SOCKET_ERROR) {
        std::cerr << "Connection failed with error: " << WSAGetLastError() << std::endl;
        closesocket(sock);
        WSACleanup();
        return 1;
    }

    std::cout << "Connected to the server!" << std::endl;

    // Send a message to the server
    const char *message = "Hello from C++ client!";
    send(sock, message, strlen(message), 0);
    std::cout << "Message sent: " << message << std::endl;

    // Close the socket
    closesocket(sock);

    // Clean up Winsock
    WSACleanup();

    return 0;
}
