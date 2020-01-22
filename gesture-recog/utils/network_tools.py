import zmq
import numpy as np
import cv2
import base64
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Dict


def concurrent_send(socks: Dict[str, zmq.Socket], msg: str) -> None:
    """Function which enables the concurrent communication with both slaves
        :param socks: dictionary containing the two sockets for the two slaves, identified by a label ('L'/'R')
        :param msg: string containing the message"""
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(socks['L'].send_string, msg)
        executor.submit(socks['R'].send_string, msg)


# noinspection PyUnresolvedReferences
def recv_frame(sock: zmq.Socket) -> np.ndarray:
    # Read frame as a base64 string and return it
    serial_frame = sock.recv_string()
    buffer = base64.b64decode(serial_frame)
    frame = cv2.imdecode(np.fromstring(buffer, dtype=np.uint8), 1)

    return frame
