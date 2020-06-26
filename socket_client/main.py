import socket
import time
from colorsys import hsv_to_rgb
from PIL import Image

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ("172.20.10.4", 8090)
s.connect(server_address)

image = b''

while True:
    while True:
        #message = input("Mensaje: ")
        #s.send((message + "\n\r").encode("utf-8"))
        #s.send("subi dubi dubi du wap \n\r".encode("utf-8"))
        while True:
            image += s.recv(1024)
            if len(image) >= 921600:
                break
        print(len(image))
        # Convert list to bytes
        if len(image) > 1000:
            img = Image.frombytes('RGB', (640, 480), image)
            img.show()
        break
    break


    print("Closing connection")
