import threading
import socket

class RegMode(threading.Thread):

    def __init__(self, host, port):
        super().__init__()
        server_address = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(server_address)
        self.reset()


    def reset(self):
        self.Register_mode = False
        self.Register_id = 0

    def run(self):
        self.sock.listen()

