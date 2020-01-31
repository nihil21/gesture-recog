import sys
import argparse
import zmq
import cv2
import base64
import time
try:
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    print("PiCamera will be used.")
    WEBCAM = False
except OSError:
    print("PiCamera not supported by the system, webcam will be used.")
    WEBCAM = True

# Camera size
CAMERA_RESOLUTION = (640, 480)
CAMERA_RESIZE = (240, 192)


def stream_from_picamera(sock: zmq.Socket, flip: bool) -> None:
    print('Streaming from PiCamera...')

    # Initialize camera
    camera = PiCamera()
    camera.resolution = CAMERA_RESIZE
    camera.framerate = 32
    raw_capture = PiRGBArray(camera, size=CAMERA_RESIZE)

    # Camera warm-up
    time.sleep(2.0)

    # Send ready signal to master
    sock.send_string('\1')

    # Wait for the starting signal
    sock.recv_string()

    for capture in camera.capture_continuous(raw_capture, format='bgr', use_video_port=True):
        # Grab raw NumPy array representing the frame
        frame = capture.array

        # Flip image, if specified
        if flip:
            frame = cv2.flip(frame, 0)

        # Send the frame as a base64 string
        encoded, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer)
        sock.send(jpg_as_text)

        raw_capture.truncate(0)

        # Try to read the termination signal from a non-blocking recv
        try:
            # If the recv succeeds, break from the loop
            # noinspection PyUnresolvedReferences
            sig = sock.recv_string(flags=zmq.NOBLOCK)
            print('Termination signal received:', sig)
            break
        except zmq.Again:
            pass
    # Release resource
    camera.close()


def stream_from_webcam(sock: zmq.Socket, flip: bool) -> None:
    print('Streaming from webcam...')

    # Initialize camera
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])

    # Camera warm-up
    time.sleep(0.1)

    # Wait for the starting signal
    sock.recv_string()

    while True:
        # Grab frame from video
        ret, frame = video_capture.read()

        # Flip image, if specified
        if flip:
            frame = cv2.flip(frame, 0)

        # Send the frame as a base64 string
        encoded, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer)
        sock.send(jpg_as_text)

        # Try to read the termination signal from a non-blocking recv
        try:
            # If the recv succeeds, break from the loop
            # noinspection PyUnresolvedReferences
            sig = sock.recv_string(flags=zmq.NOBLOCK)
            print('Termination signal received', sig)
            break
        except zmq.Again:
            pass
    # Release resource
    video_capture.release()


def shot_from_picamera(sock: zmq.Socket, flip: bool) -> None:
    print('Taking a picture from PiCamera...')

    # Initialize camera
    camera = PiCamera()
    camera.resolution = CAMERA_RESOLUTION
    camera.framerate = 20
    raw_capture = PiRGBArray(camera, size=CAMERA_RESOLUTION)

    # Camera warm-up
    time.sleep(0.1)

    # Tell the master that the camera is ready
    sock.send_string('Ready')
    print(sock.recv_string())

    camera.capture(raw_capture, format='bgr')
    # Grab raw NumPy array representing the frame
    frame = raw_capture.array

    # Flip image, if specified
    if flip:
        frame = cv2.flip(frame, 0)

    # Send the frame as a base64 string
    encoded, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer)
    sock.send(jpg_as_text)

    # Release resource
    camera.close()


def shot_from_webcam(sock: zmq.Socket, flip: bool) -> None:
    print('Taking a picture from webcam...')

    # Initialize camera
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])

    # Camera warm-up
    time.sleep(0.1)

    # Tell the master that the camera is ready
    sock.send_string('Ready')
    print(sock.recv_string())

    # Grab frame from video
    ret, frame = video_capture.read()
    frame = cv2.resize(frame, CAMERA_RESOLUTION)

    # Flip image, if specified
    if flip:
        frame = cv2.flip(frame, 0)

    # Send the frame as a base64 string
    encoded, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer)
    sock.send(jpg_as_text)

    # Release resource
    video_capture.release()


def main():
    # Construct argument parser and add arguments to it
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip_address", required=True, help="hostname of the master")
    ap.add_argument("-p", "--port", required=True, help="port on which the slave connects to master")
    ap.add_argument("-f", "--flip", action="store_true", help="if set, image is flipped before it is sent to master")
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
    flip = args['flip']

    context = None
    sock = None
    try:
        # Connect to master
        context = zmq.Context()
        # noinspection PyUnresolvedReferences
        sock = context.socket(zmq.PAIR)
        # noinspection PyUnresolvedReferences
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect("tcp://{}:{}".format(ipaddr, port))

        # Confirm connection of the slave itself
        print(sock.recv_string())

        # Confirm connection of the other slave
        print(sock.recv_string())

        # User input cycle
        while True:
            print('Waiting for user input...')
            # Read user's choice
            sel = int(sock.recv_string())
            if sel == 1 or sel == 3:
                # Start streaming
                stream = stream_from_picamera if not WEBCAM else stream_from_webcam
                stream(sock, flip)
            # elif sel == 3:
            #    shot = shot_from_picamera if not WEBCAM else shot_from_webcam
            #    shot(sock, flip)
            if sel == 4:
                break
    except KeyboardInterrupt:
        print('')
        print('Enforcing termination manually')
    finally:
        # Closing sockets
        sock.close()
        context.term()
        print('Terminating...')


main()
