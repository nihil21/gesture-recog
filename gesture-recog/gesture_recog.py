import zmq
from concurrent.futures import ThreadPoolExecutor
from utils.calibration_tools import capture_images, calibrate, disp_map
from utils.network_tools import concurrent_send

# Ports for both cameras
DX_PORT = 8000
SX_PORT = 8001

# Folders to store images for calibration
DX_IMG_FOLDER = "../calibration-images/dx/"
SX_IMG_FOLDER = "../calibration-images/sx/"

# Chessboard size
PATTERN_SIZE = (8, 5)

# Square length
SQUARE_LEN = 26.5  # mm

# Folders to store calibration data
DX_CALIB_FOLDER = "../calibration-data/dx/"
SX_CALIB_FOLDER = "../calibration-data/sx/"


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


def accept_client_thread(sock: zmq.Socket, sock_idx: str) -> None:
    """Confirms connection to client by sending a message
        :param sock: the zmq socket
        :param sock_idx: the index of the client ('DX'/'SX')"""

    sock.send_string('Connection established with server')
    print('Connection established with client {}'.format(sock_idx))


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


def main():
    context = None
    socks = None
    try:
        # Set up zmq context and sockets PAIR
        context = zmq.Context()
        socks = {'DX': create_socket(context, DX_PORT), 'SX': create_socket(context, SX_PORT)}
        print('Waiting on ports {} and {}...'.format(DX_PORT, SX_PORT))

        # Set up other environment variables
        img_folders = {'DX': DX_IMG_FOLDER, 'SX': SX_IMG_FOLDER}
        calib_folders = {'DX': DX_CALIB_FOLDER, 'SX': SX_CALIB_FOLDER}

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

            # Tell clients to prepare for streaming, by sending user's selection,
            # unless he chose calibration (done server-side only)
            if sel != 2 and sel != 3:
                concurrent_send(socks, str(sel))
            if sel == 4:
                break

            # Invoke the corresponding function
            if sel == 1:
                capture_images(socks, img_folders)
            elif sel == 2:
                calibrate(img_folders, PATTERN_SIZE, SQUARE_LEN, calib_folders)
            elif sel == 3:
                disp_map()
    except KeyboardInterrupt:
        print('')
        print('Enforcing termination manually')
    finally:
        # Closing sockets
        for sock in socks.values():
            sock.close()
        context.term()
        print('Terminating...')


main()
