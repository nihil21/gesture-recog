import sys
import argparse
import zmq
import cv2
import base64
import time
# from picamera.array import PiRGBArray
# from picamera import PiCamera

# Camera width and height
CW = 640
CH = 480


def capture_from_webcam(sock: zmq.Socket):
    # Initialize camera
    video_capture = cv2.VideoCapture(0)

    # Camera warm-up
    time.sleep(0.1)

    while True:
        # Grab frame from video
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, (CW, CH))
        encoded, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer)
        sock.send(jpg_as_text)

        # Try to read the termination signal from a non-blocking recv
        try:
            # If the recv succeeds, break from the loop
            sock.recv_string(flags=zmq.NOBLOCK)
            print('Termination signal received')
            break
        except zmq.Again:
            pass

    video_capture.release()


def capture_from_picamera(sock: zmq.Socket):
    # Initialize camera
    '''camera = PiCamera()
    camera.resolution = (CW, CH)
    camera.framerate = 32
    raw_capture = PiRGBArray(camera, size=(CW, CH))'''

    # Camera warm-up
    time.sleep(0.1)

    '''for capture in camera.capture_continuous(raw_capture, format='bgr', use_video_port=True):
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
            break'''


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
        msg = sock.recv_string()
        print(msg)

        # Confirm connection of the other client
        msg = sock.recv_string()
        print(msg)

        capture_from_webcam(sock)
    finally:
        # Closing sockets
        sock.close()
        context.term()
        print('Terminating...')


main()
