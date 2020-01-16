import cv2
import numpy as np
import zmq
import base64
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Callable

# Ports for both cameras
DX_PORT = 8000
SX_PORT = 8001

# Folders to store images for calibration
DX_FOLDER = "../calibration/dx/"
SX_FOLDER = "../calibration/sx/"

# Chessboard size
BS = (8, 5)


# noinspection PyUnresolvedReferences
def create_socket(context: zmq.Context, tcp_port: int) -> zmq.Socket:
    """Creates a zmq.PAIR socket
        :param context: the zmq context
        :param tcp_port: integer representing the port

        :return sock: the zmq.PAIR socket created"""

    sock = context.socket(zmq.PAIR)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind('tcp://*:{:d}'.format(tcp_port))

    return sock


def accept_client_thread(sock: zmq.Socket, sock_idx: str):
    """Confirms connection to client by sending a message
        :param sock: the zmq socket
        :param sock_idx: the index of the client ('DX'/'SX')"""

    sock.send_string('Connection established with server')
    print('Connection established with client {}'.format(sock_idx))


def concurrent_send(socks: Dict[str, zmq.Socket], msg: str):
    with ThreadPoolExecutor() as executor:
        executor.submit(socks['DX'].send_string, msg)
        executor.submit(socks['SX'].send_string, msg)


def user_input() -> int:
    """Displays a menu of the available actions and asks user's input
        :return sel: integer representing user's choice"""

    print('')
    print('=' * 40)
    print('1 - Collect images for calibration')
    print('2 - Perform calibration')
    print('3 - Real time disparity map')
    print('4 - Exit')

    while True:
        try:
            sel = input('Select one of the options [1, 2, 3, 4]: ')
            sel = int(sel)
            if sel not in [1, 2, 3, 4]:
                print('The option inserted is not valid, retry.')
            else:
                break
        except ValueError:
            print('The option inserted is not numeric, retry.')
    print('-' * 40)
    return sel


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


def capture_images(socks: Dict[str, zmq.Socket]):
    print('Collecting images of a chessboard for calibration...')
    
    # Receive confirmation by the client and send signal to synchronize both cameras
    with ThreadPoolExecutor() as executor:
        executor.submit(socks['DX'].recv_string)
        executor.submit(socks['SX'].recv_string)
    
    # Receive frames by both clients using threads
    print("Both cameras are ready")
    with ThreadPoolExecutor() as executor:
        executor.submit(capture_images_thread, socks['DX'], 'DX')
        executor.submit(capture_images_thread, socks['SX'], 'SX')
    print('Images collected')


# noinspection PyUnresolvedReferences
def capture_images_thread(sock: zmq.Socket, sock_idx: str):
    sock.send_string("Start signal received")

    # Initialize variables for countdown
    n_pics, tot_pics = 0, 5
    n_sec, tot_sec = 0, 4
    str_sec = '4321'
    start_time = datetime.now()

    # Loop until 'tot_pics' images are collected
    while n_pics < tot_pics:
        # Read frame as a base64 string
        serial_frame = sock.recv_string()
        buffer = base64.b64decode(serial_frame)
        frame = cv2.imdecode(np.fromstring(buffer, dtype=np.uint8), 1)

        # Display counter on screen before saving frame
        if n_sec < tot_sec:
            # Draw on screen the current remaining seconds
            frame = cv2.putText(img=frame,
                                text=str_sec[n_sec],
                                org=(int(40), int(80)),
                                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=3,
                                color=(255, 255, 255),
                                thickness=5,
                                lineType=cv2.LINE_AA)

            # If time elapsed is greater than one second, update 'n_sec'
            time_elapsed = (datetime.now() - start_time).total_seconds()
            if time_elapsed >= 1:
                n_sec += 1
                start_time = datetime.now()
        else:
            # When countdown ends, save image to file
            path = DX_FOLDER if sock_idx == 'DX' else SX_FOLDER
            cv2.imwrite(path + '{:02d}'.format(n_pics) + '.jpg', frame)
            n_pics += 1
            n_sec = 0
            print('{}: {:d}/{:d} images collected'.format(sock_idx, n_pics, tot_pics))

        cv2.imshow('{} frame'.format(sock_idx), frame)
        cv2.waitKey(1)  # invocation of non-blocking 'waitKey', required by OpenCV after 'imshow'
        if n_pics == tot_pics:
            # If enough images are collected, the termination signal is sent to the client
            sock.send_string('\0')

    cv2.destroyAllWindows()


# TODO
def calibrate():
    pass


# TODO
def disp_map():
    pass


def main():
    context = None
    socks = None
    try:
        # Set up zmq context and sockets PAIR
        context = zmq.Context()
        socks = {'DX': create_socket(context, DX_PORT), 'SX': create_socket(context, SX_PORT)}
        print('Waiting on ports {} and {}...'.format(DX_PORT, SX_PORT))

        # Accept connections in a thread pool
        with ThreadPoolExecutor() as executor:
            executor.submit(accept_client_thread, socks['DX'], 'DX')
            executor.submit(accept_client_thread, socks['SX'], 'SX')

        # Confirm connection to both clients by sending a message
        msg = 'Connection established with both clients'
        print(msg)
        concurrent_send(socks, msg)

        # User input cycle
        while True:
            # Display action menu and ask for user input
            sel = user_input()
            # Send user's choice to both clients
            concurrent_send(socks, str(sel))
            if sel == 4:
                break
            # Select the corresponding function
            f = select_function(sel)
            f(socks)

    finally:
        # Closing sockets
        for sock in socks.values():
            sock.close()
        context.term()
        print('Terminating...')


main()
