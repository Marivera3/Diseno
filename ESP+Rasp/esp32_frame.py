import socket
from PIL import Image
import io
import numpy as np

def esp32_frame(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    ip = socket.gethostbyname(host)
    server_address = (ip, port)
    try:
        s.connect(server_address)
        image = b''
        buf_len = int.from_bytes(s.recv(2), "little")
    except:
        return None;

    while True:
        image += s.recv(1024)
        if len(image) >= buf_len:
            break
    if len(image) > 1000:
        img = Image.open(io.BytesIO(image))
        frame = np.array(img)
        return img

if __name__ == "__main__":
    esp32_frame("190.162.132.149", 1228)
