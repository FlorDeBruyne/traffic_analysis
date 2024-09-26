import socket
import os
import threading
from dotenv import load_dotenv, dotenv_values

load_dotenv()

HOST = os.getenv("SERVER_ADDRESS")
PORT = os.getenv("PORT")
ADDR = (socket.gethostbyname(socket.gethostname()), int(PORT))

class Client():

    def __init__(self) -> None:
        print("[STARTING] Cient is starting\n")
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect(ADDR)
    
    def data_transfer(self, data) -> bool:
        self.client_socket.send(data.encode())

        if not self.client_socket.recv(2048).decode():
            self.client_socket.close()
            return False
        
        self.client_socket.close()
        return True


