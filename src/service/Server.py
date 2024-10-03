import socket, os, threading
from dotenv import load_dotenv, dotenv_values

load_dotenv()


HOST = os.getenv("SERVER_ADDRESS")
PORT = os.getenv("PORT")
ADD = (socket.gethostbyname(socket.gethostname()), int(PORT))
SERVER_FOLDER = "/home/flor/data/" #os.getenv("SERVER_FOLDER")

print("[STARTING] Server is starting\n")
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADD)


def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")

    file = open('traffic_data.zip', "wb")

    connected = True
    while connected:
        data = conn.recv(2048)
        print("[SERVER] Recieving data")
        if data == b"done":
            conn.close()
        
        file.write(data)
        
    file.close()
    print("[SERVER] Transfering file")
    placement = open(os.path.join(SERVER_FOLDER, file), "w")

    conn.send("received")

    conn.close()


def start():
    #listen for incoming connections from clients, maximum queue of 2 clients
    server.listen(2)

    while True:
        connection, address = server.accept()
        thread = threading.Thread(target=handle_client, args=(connection, address))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.active_count() - 1}")
        
        
start()










