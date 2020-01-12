import sys
import argparse
import socket
import cv2
import pickle
import time
from picamera.array import PiRGBArray
from picamera import PiCamera

# Camera width and height
CW = 640
CH = 480


def main():
    # Construct argument parser and add arguments to it
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip_address", required=True, help="hostname of the server")
    ap.add_argument("-p", "--port", required=True, help="port on which the client connects to server")
    args = vars(ap.parse_args())

    # Argument reading and check
    ipaddr = args['ip_address']
    port = args['port']
    try:
        port = int(port)
    except ValueError:
        sys.exit("Argument 'port' must be an integer.")
    if not 1024 <= port <= 65535:
        sys.exit("Argument 'port' must within range [1024, 65535].")

    # Connect to server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ipaddr, port))

        # Confirm connection of the client itself
        msg = sock.recv(4096).decode(encoding='utf-8')
        print(msg)

        # Confirm connection of the other client
        msg = sock.recv(4096).decode(encoding='utf-8')
        print(msg)

        # Initialize camera
        camera = PiCamera()
        camera.resolution = (CW, CH)
        camera.framerate = 32
        raw_capture = PiRGBArray(camera, size=(CW, CH))

        # Camera warm-up
        time.sleep(0.1)

        for capture in camera.capture_continuous(raw_capture, format='bgr', use_video_port=True):
            # Grab raw NumPy array representing the frame
            frame = capture.array

            # Send the serialized frame and the termination sequence '\term'
            serial_frame = pickle.dumps(frame)
            sock.send(serial_frame)
            sock.send(b'\term')

            # Clear stream for next frame
            raw_capture.truncate(0)

            # Read confirmation
            confirm = sock.recv(4096).decode(encoding='utf-8')
            print(confirm)
            if confirm == 'Stop':
                break


main()
