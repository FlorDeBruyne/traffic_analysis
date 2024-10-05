import socket
import os
import threading
from dotenv import load_dotenv, dotenv_values

load_dotenv()

HOST = os.getenv("SERVER_ADDRESS")
PORT = os.getenv("PORT")
ADDR = (socket.gethostbyname(socket.gethostname()), int(PORT))

def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

class Client():

    def __init__(self) -> None:
        print("[STARTING] Client is starting\n")
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect(ADDR)
    
    @threaded
    def data_transfer(self, data_location) -> bool:
        data = open(data_location, 'rb')
        loading = data.read(1024)

        while loading:
            self.client_socket.send(loading)
            print("[CLIENT] Sending data")
            loading = data.read(1024)

        self.client_socket.send(b"done")
        
        if not self.client_socket.recv(1024):
            print("[ERROR] Tranfer not complete, retrying")
            self.data_transfer(data_location)
        
        print("[CLIENT] Transfer complete, disconnecting")
        self.client_socket.close()
        return True


