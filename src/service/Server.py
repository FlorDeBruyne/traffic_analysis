import socket, os, threading
from dotenv import load_dotenv, dotenv_values

load_dotenv()


HOST = os.getenv("SERVER_ADDRESS")
PORT = os.getenv("PORT")
ADD = (socket.gethostbyname(socket.gethostname()), PORT)
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADD)

def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    while connected:
        data = conn.recv(2048).decode()
        
    conn.send("received".encode())
    conn.close()


def start():
    #listen for incoming connections from clients, maximum queue of 2 clients
    server.listen(2)

    while True:
        connection, address = server.accept()
        thread = threading.Thread(target=handle_client, args=(connection, address))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")
        
        
start()










