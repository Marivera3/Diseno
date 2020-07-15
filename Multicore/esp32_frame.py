import socket
from PIL import Image
import io

def esp32_frame(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    server_address = (host, port)
    try:
        s.connect(server_address)
    except socket.timeout:
        return None;

    image = b''
    buf_len = int.from_bytes(s.recv(2), "little")
    print(buf_len)
    while True:
        image += s.recv(1024)
        if len(image) >= buf_len:
            break
    # Convert list to bytes
    if len(image) > 1000:
        img = Image.open(io.BytesIO(image))
        return img

if __name__ == "__main__":
    esp32_frame("172.20.10.4", 8090)
