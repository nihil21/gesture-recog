import zmq
import numpy as np
import cv2
import base64
import time
from typing import Tuple, Optional
try:
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    PICAM: bool = True
except OSError:
    PICAM: bool = False


class NetworkAgent:
    """Class representing a generic 'NetworkAgent', able to bind/connect and to communicate over a TCP socket.

    Attributes:
        context -- the ZMQ context underlying the socket
        sock --- the ZMQ TCP socket over which the communication is performed

    Methods:
        send_msg --- method enabling the sending of a string message over a TCP socket
        recv_msg --- method enabling the reception of a string message over a TCP socket
        send_frame --- method enabling the sending of an OpenCV image/NumPy array over a TCP socket
        recv_frame --- method enabling the reception of an OpenCV image/NumPy array over a TCP socket
        close --- method which releases the network resources used by the object"""

    def __init__(self,
                 port: int,
                 ip_addr: Optional[str] = None):
        """Initialize the NetworkAgent object by setting the ZMQ context, a zmq.PAIR socket and by performing the
        binding to the given port or the connect to the given address and port, depending on the parameters
            :param port: integer representing the port to which the socket will be bound
            :param ip_addr: optional string representing the ip address to connect to; if set to 'None' (default),
            the 'NetworkAgent' binds to the given port, otherwise it connects to the given address and port"""

        # Set up context and socket with zero linger time
        self.context: zmq.Context = zmq.Context()
        # noinspection PyUnresolvedReferences
        self.sock: zmq.Socket = self.context.socket(zmq.PAIR)
        # noinspection PyUnresolvedReferences
        self.sock.setsockopt(zmq.LINGER, 0)

        # Perform the bind or the connect depending on the 'ip_addr' parameter
        if ip_addr is None:
            self.sock.bind('tcp://*:{0:d}'.format(port))
        else:
            self.sock.connect('tcp://{:s}:{:d}'.format(ip_addr, port))

    def send_msg(self, msg: str):
        """Method implementing the sending of a message over a TCP socket
            :param msg: string representing the message to be sent"""
        self.sock.send_string(msg)

    def recv_msg(self) -> str:
        """Method implementing the reception of a message over a TCP socket
            :returns a string representing the message received"""
        return self.sock.recv_string()

    def send_frame(self, frame: np.ndarray):
        """Method that enables an OpenCV image/NumPy array to be sent over a TCP socket
            :param frame: the OpenCV image/NumPy array to be serialized and sent over the socket"""
        encoded, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer)
        self.sock.send(jpg_as_text)

    def recv_frame(self) -> np.ndarray:
        """Method that enables a OpenCV image/NumPy array to be received through a TCP socket
            :return the reconstructed OpenCV image/NumPy array received through the socket"""
        serial_frame = self.sock.recv_string()
        buffer = base64.b64decode(serial_frame)
        frame = cv2.imdecode(np.fromstring(buffer, dtype=np.uint8), 1)
        return frame

    def close(self):
        """Method that closes the socket and the context to free resources"""
        self.sock.close()
        self.context.term()


class ImageSender(NetworkAgent):
    """Class representing an ImageSender which binds to a given port and streams the images captured
    over a TCP socket to an ImageReceiver.

    Arguments:
        res --- the resolution at which the images are captured and sent over the underlying TCP socket
        flip --- a parameter indicating whether to flip the image before being sent

    Methods:
        stream --- the method implementing the streaming of images which are captured by the device and sent over the
        underlying TCP socket"""

    def __init__(self,
                 port: int,
                 res: Tuple[int, int],
                 flip: Optional[bool] = False):
        """Initializes the ImageSender object by calling the 'NetworkAgent' __init__.py, so that it binds to a
        given port. Moreover, it sets the type of camera from which the streaming is performed (Webcam or PiCamera),
        and whether the image must be flipped or not (useful for particular hardware setups)
            :param port: integer representing the port to which the socket will be bound
            :param res: tuple of two ints representing the resolution of the camera
            :param flip: optional boolean, if 'True' the image will be flipped before being sent (by default
            it is set to 'False')"""

        super().__init__(port=port)

        # Set camera type
        if PICAM:
            self.stream = self.stream_from_picamera
        else:
            self.stream = self.stream_from_webcam
        # Set resolution
        self.res = res
        # Set flip option
        self.flip = flip

    def recv_frame(self):
        # Disable 'recv_frame' method
        raise NotImplementedError("The recv_frame method of ImageSender class is disabled")

    def stream_from_picamera(self):
        """Function that implements the streaming of images captured by the PiCamera over a TCP socket"""
        print('Streaming from PiCamera...')

        # Initialize camera
        camera = PiCamera()
        camera.resolution = self.res
        camera.framerate = 10
        raw_capture = PiRGBArray(camera, size=self.res)

        # Camera warm-up
        time.sleep(2.0)

        # Send ready message to master and wait for the starting signal
        self.send_msg('ready')
        sig = self.recv_msg()
        print('Master: {0:s}'.format(sig))

        for capture in camera.capture_continuous(raw_capture, format='bgr', use_video_port=True):
            # Grab raw NumPy array representing the frame
            frame = capture.array

            # Flip image, if specified
            if self.flip:
                frame = cv2.flip(frame, 0)

            # Send the frame
            self.send_frame(frame)

            raw_capture.truncate(0)

            # Try to read the termination signal from a non-blocking recv
            try:
                # If the recv succeeds, break from the loop
                # noinspection PyUnresolvedReferences
                sig = self.sock.recv_string(flags=zmq.NOBLOCK)
                print('Master: {0:s}'.format(sig))
                break
            except zmq.Again:
                pass
        # Release resource
        camera.close()
        print('End transmission')

    def stream_from_webcam(self):
        """Function that implements the streaming of images captured by the webcam over a TCP socket"""
        print('Streaming from webcam...')

        # Initialize camera
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.res[0])
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.res[1])

        # Camera warm-up
        time.sleep(0.1)

        # Send ready message to master and wait for the starting signal
        self.send_msg('ready')
        sig = self.recv_msg()
        print('Master: {0:s}'.format(sig))

        while True:
            # Grab frame from video
            ret, frame = video_capture.read()

            # Flip image, if specified
            if self.flip:
                frame = cv2.flip(frame, 0)

            # Send the frame
            self.send_frame(frame)

            # Try to read the termination signal from a non-blocking recv
            try:
                # If the recv succeeds, break from the loop
                # noinspection PyUnresolvedReferences
                sig = self.sock.recv_string(flags=zmq.NOBLOCK)
                print('Master: {0:s}'.format(sig))
                break
            except zmq.Again:
                pass
        # Release resource
        video_capture.release()
        print('End transmission')


class ImageReceiver(NetworkAgent):
    """Class representing an ImageReceiver which connects to a given address and port and receives the images captured
    by an ImageSender.

    Methods:
        flush_pending_frames --- method that flushes the pending frames on the TCP socket"""

    def __init__(self,
                 ip_addr: str,
                 port: int):
        super().__init__(port=port, ip_addr=ip_addr)

    def send_frame(self, array: np.ndarray):
        # Disable 'send_frame' method
        raise NotImplementedError("The send_frame method of ImageReceiver class is disabled")

    def flush_pending_frames(self):
        """Method that flushes the pending frames on the TCP socket"""
        # noinspection PyUnresolvedReferences
        self.sock.setsockopt(zmq.RCVTIMEO, 1000)
        while True:
            # When recv timeout expires, break from the loop
            try:
                self.recv_frame()
            except zmq.Again:
                break
        # noinspection PyUnresolvedReferences
        self.sock.setsockopt(zmq.RCVTIMEO, -1)
