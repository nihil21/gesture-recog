import zmq
import numpy as np
import cv2
import base64
from typing import Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


# noinspection PyUnresolvedReferences
def create_socket_bind(context: zmq.Context, tcp_port: int) -> zmq.Socket:
    """Creates a zmq.PAIR socket and binds it to the given port
        :param context: the zmq context
        :param tcp_port: integer representing the port

        :return sock: the zmq.PAIR socket created"""

    sock = context.socket(zmq.PAIR)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind('tcp://*:{:d}'.format(tcp_port))

    return sock


# noinspection PyUnresolvedReferences
def create_socket_connect(context: zmq.Context, ip_address: str, tcp_port: int) -> zmq.Socket:
    """Creates a zmq.PAIR socket and connects it to the given address and port
        :param context: the zmq context
        :param ip_address: address to connect to
        :param tcp_port: integer representing the port

        :return sock: the zmq.PAIR socket created"""

    sock = context.socket(zmq.PAIR)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect('tcp://{:s}:{:d}'.format(ip_address, tcp_port))

    return sock


def concurrent_send(socks: Dict[str, zmq.Socket], msg: str) -> None:
    """Function which enables the concurrent communication with both sensors
        :param socks: dictionary containing the two sockets for the two sensors, identified by a label ('L'/'R')
        :param msg: string containing the message"""
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(socks['L'].send_string, msg)
        executor.submit(socks['R'].send_string, msg)


def concurrent_recv(socks: Dict[str, zmq.Socket]) -> None:
    """Function which enables the concurrent signal reception from both sensors
        :param socks: dictionary containing the two sockets for the two sensors, identified by a label ('L'/'R')"""
    with ThreadPoolExecutor(max_workers=2) as executor:
        futureL = executor.submit(socks['L'].recv_string)
        futureR = executor.submit(socks['R'].recv_string)
        for future in as_completed([futureL, futureR]):
            if future == futureL:
                print('L: {}'.format(futureL.result()))
            else:
                print('R: {}'.format(futureR.result()))


def send_frame(sock: zmq.Socket, frame: np.ndarray) -> None:
    """Function that sends a NumPy array as a base64 string
        :param sock: zmq socket on which the array is sent
        :param frame: the array to be sent"""
    encoded, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer)
    sock.send(jpg_as_text)


def recv_frame(sock: zmq.Socket) -> np.ndarray:
    """Function that implements the receiving of a NumPy array
        :param sock: zmq socket on which the array is received

        :return frame: the array received from the socket"""
    serial_frame = sock.recv_string()
    buffer = base64.b64decode(serial_frame)
    frame = cv2.imdecode(np.fromstring(buffer, dtype=np.uint8), 1)
    return frame


def concurrent_recv_frame(socks: Dict[str, zmq.Socket]) -> Tuple[np.ndarray, np.ndarray]:
    """Function that implements the concurrent receiving of two NumPy arrays from two sensors
        :param socks: dictionary containing the two sockets for the two sensors, identified by a label ('L'/'R')

        :return frameL: the array received from the left sensor
        :return frameR: the array received from the right sensor"""
    with ThreadPoolExecutor(max_workers=2) as executor:
        futureL = executor.submit(recv_frame, socks['L'])
        futureR = executor.submit(recv_frame, socks['R'])
        for future in as_completed([futureL, futureR]):
            if future == futureL:
                frameL = future.result()
            else:
                frameR = future.result()
    return frameL, frameR


def concurrent_flush(socks: Dict[str, zmq.Socket]) -> None:
    """Function that flushes concurrently the two zmq sockets
        :param socks: dictionary containing the two zmq sockets for the two sensors, identified by a label"""
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(flush, socks['L'])
        executor.submit(flush, socks['R'])
    print('Sockets flushed')


# noinspection PyUnresolvedReferences
def flush(sock: zmq.Socket) -> None:
    """Function that flushes a zmq socket
        :param sock: zmq socket to flush"""
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    while True:
        # When recv timeout expires, break from the loop
        try:
            sock.recv_string()
        except zmq.Again:
            break
    sock.setsockopt(zmq.RCVTIMEO, -1)
