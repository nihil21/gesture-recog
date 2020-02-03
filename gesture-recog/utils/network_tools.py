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


# noinspection PyUnresolvedReferences
def recv_frame(sock: zmq.Socket, camera_idx: str) -> Tuple[np.ndarray, str]:
    # Read frame as a base64 string and return it
    serial_frame = sock.recv_string()
    buffer = base64.b64decode(serial_frame)
    frame = cv2.imdecode(np.fromstring(buffer, dtype=np.uint8), 1)
    return frame, camera_idx


# noinspection PyUnresolvedReferences
def send_frame(sock: zmq.Socket, frame: np.ndarray) -> None:
    encoded, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer)
    sock.send(jpg_as_text)


def concurrent_flush(socks: Dict[str, zmq.Socket])-> None:
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(flush, socks['L'])
        executor.submit(flush, socks['R'])
    print('Sockets flushed')


# noinspection PyUnresolvedReferences
def flush(sock: zmq.Socket) -> None:
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    while True:
        # When recv timeout expires, break from the loop
        try:
            sock.recv_string()
        except zmq.Again:
            break
    sock.setsockopt(zmq.RCVTIMEO, -1)
