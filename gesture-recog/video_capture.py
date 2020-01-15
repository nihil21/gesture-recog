import sys
import argparse
import zmq
import cv2
import base64
import time
from typing import Dict, Callable
#from picamera.array import PiRGBArray
#from picamera import PiCamera

# Camera size
CS = (640, 480)


def select_function(sel: int) -> Callable[[Dict[str, zmq.Socket]], None]:
    """Selects the function corresponding to user's choice
        :param sel: integer representing user's choice

        :return f: function corresponding to user's choice"""
    # Switcher dictionary associating a number to a function
    switcher = {
        1: capture_images,
        2: calibrate,
        3: disp_map
    }
    # Get function from switcher dictionary
    f = switcher.get(sel)
    return f


def capture_images(sock):
    print('Collecting images for calibration...')

    # Initialize camera
    video_capture = cv2.VideoCapture(0)

    # Camera warm-up
    time.sleep(0.1)

    while True:
        # Grab frame from video
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, CS)

        # Send the frame as a base64 string
        encoded, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer)
        sock.send(jpg_as_text)

        # Try to read the termination signal from a non-blocking recv
        try:
            # If the recv succeeds, break from the loop
            sig = sock.recv_string(flags=zmq.NOBLOCK)
            print('Termination signal received', sig)
            break
        except zmq.Again:
            pass

    video_capture.release()
    print('Images collected')


# TODO
def calibrate():
    pass


# TODO
def disp_map():
    pass


def capture_from_picamera(sock: zmq.Socket):
    # Initialize camera
    #camera = PiCamera()
    #camera.resolution = (CW, CH)
    #camera.framerate = 32
    #raw_capture = PiRGBArray(camera, size=(CW, CH))

    # Camera warm-up
    time.sleep(0.1)

    #for capture in camera.capture_continuous(raw_capture, format='bgr', use_video_port=True):
        # Grab raw NumPy array representing the frame
    #    frame = capture.array

        # Send the frame as a base64 string
    #    encoded, buffer = cv2.imencode('.jpg', frame)
    #    jpg_as_text = base64.b64encode(buffer)
    #    sock.send(jpg_as_text)

        # Clear stream for next frame
    #    raw_capture.truncate(0)

        # Try to read the termination signal from a non-blocking recv
    #    try:
            # If the recv succeeds, break from the loop
    #        sig = sock.recv_string(flags=zmq.NOBLOCK)
    #        print('Termination signal received', sig)
    #        break
    #    except zmq.Again:
    #        pass


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

    try:
        # Connect to server
        context = zmq.Context()
        sock = context.socket(zmq.PAIR)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect("tcp://{}:{}".format(ipaddr, port))

        # Confirm connection of the client itself
        print(sock.recv_string())

        # Confirm connection of the other client
        print(sock.recv_string())

        # User input cycle
        while True:
            print('Waiting for user input...')
            # Read user's choice
            sel = int(sock.recv_string())
            if sel == 4:
                break
            # Select corresponding function
            f = select_function(sel)
            f(sock)

    finally:
        # Closing sockets
        sock.close()
        context.term()
        print('Terminating...')


main()
