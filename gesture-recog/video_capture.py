import sys
import argparse
import zmq
import cv2
import time
import utils.network_tools as nt
try:
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    print("PiCamera will be used.")
    WEBCAM = False
except OSError:
    print("PiCamera not supported by the system, webcam will be used.")
    WEBCAM = True

# Ports of the sensors
L_PORT = 8000
R_PORT = 8001

# Camera size
CAMERA_RESOLUTION = (300, 300)


def stream_from_picamera(sock: zmq.Socket, flip: bool) -> None:
    print('Streaming from PiCamera...')

    # Initialize camera
    camera = PiCamera()
    camera.resolution = CAMERA_RESOLUTION
    camera.framerate = 20
    raw_capture = PiRGBArray(camera, size=CAMERA_RESOLUTION)

    # Camera warm-up
    time.sleep(2.0)

    # Send ready message to master and wait for the starting signal
    sock.send_string('ready')
    sig = sock.recv_string()
    print('Master: {}'.format(sig))

    for capture in camera.capture_continuous(raw_capture, format='bgr', use_video_port=True):
        # Grab raw NumPy array representing the frame
        frame = capture.array

        # Flip image, if specified
        if flip:
            frame = cv2.flip(frame, 0)

        # Send the frame
        nt.send_frame(sock, frame)

        raw_capture.truncate(0)

        # Try to read the termination signal from a non-blocking recv
        try:
            # If the recv succeeds, break from the loop
            # noinspection PyUnresolvedReferences
            sig = sock.recv_string(flags=zmq.NOBLOCK)
            print('Master: {}'.format(sig))
            break
        except zmq.Again:
            pass
    # Release resource
    camera.close()
    print('End transmission')


def stream_from_webcam(sock: zmq.Socket, flip: bool) -> None:
    print('Streaming from webcam...')

    # Initialize camera
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])

    # Camera warm-up
    time.sleep(0.1)

    # Send ready message to master and wait for the starting signal
    sock.send_string('ready')
    sig = sock.recv_string()
    print('Master: {}'.format(sig))

    while True:
        # Grab frame from video
        ret, frame = video_capture.read()

        # Flip image, if specified
        if flip:
            frame = cv2.flip(frame, 0)

        # Send the frame
        nt.send_frame(sock, frame)

        # Try to read the termination signal from a non-blocking recv
        try:
            # If the recv succeeds, break from the loop
            # noinspection PyUnresolvedReferences
            sig = sock.recv_string(flags=zmq.NOBLOCK)
            print('Master: {}'.format(sig))
            break
        except zmq.Again:
            pass
    # Release resource
    video_capture.release()
    print('End transmission')


def main():
    # Construct argument parser and add arguments to it
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--orientation", required=True, help="left/right orientation of the sensor ('L'/'R')")
    ap.add_argument("-f", "--flip", action="store_true", help="if set, image is flipped before it is sent to master "
                                                              "(useful only in particular hardware setups)")
    args = vars(ap.parse_args())

    # Argument reading and check
    orientation = args['orientation']
    if orientation == 'L':
        port = L_PORT
    elif orientation == 'R':
        port = R_PORT
    else:
        sys.exit("Argument 'orientation' must be either 'L' for left or 'R' for right.")
    flip = args['flip']

    context = None
    sock = None
    try:
        # Set up context and socket
        context = zmq.Context()
        sock = nt.create_socket_bind(context, port)

        """
        # Send connection message to master and wait for a confirmation
        print('Waiting on port {}...'.format(port))
        sock.send_string('connection accepted')
        print('Connection established with master, waiting for the other sensor...')
        sig = sock.recv_string()
        print('Master: {}'.format(sig))
        """

        while True:
            print('Waiting on port {}...'.format(port))
            sig = sock.recv_string()
            print('Master: {}'.format(sig))

            """
            print('Waiting for a command...')
            cmd = int(sock.recv_string())
            if cmd == 4:
                break
            """

            # Start streaming
            stream = stream_from_picamera if not WEBCAM else stream_from_webcam
            stream(sock, flip)
    except KeyboardInterrupt:
        print('\nTermination enforced manually')
    finally:
        # Closing socket and context
        sock.close()
        context.term()
        print('Terminating...')


if __name__ == '__main__':
    main()
