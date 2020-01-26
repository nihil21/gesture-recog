import zmq
from concurrent.futures import ThreadPoolExecutor
from utils.calibration_tools import capture_images, calibrate, disp_map
from utils.network_tools import concurrent_send

# Ports for both cameras
L_PORT = 8000
R_PORT = 8001

# Folders to store images for calibration
L_IMG_FOLDER = "../calibration-images/L/"
R_IMG_FOLDER = "../calibration-images/R/"

# Chessboard size
PATTERN_SIZE = (8, 5)

# Square length
SQUARE_LEN = 26.5  # mm

# Folders to store calibration data
CALIB_FILE = "../calibration-data/calib"

# Camera size
CAMERA_RESOLUTION = (640, 480)


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


def accept_slave_thread(sock: zmq.Socket, camera_idx: str) -> None:
    """Confirms connection to slave by sending a message
        :param sock: the zmq socket
        :param camera_idx: the index of the camera attached to the slave ('L'/'R')"""

    sock.send_string('Connection established with master')
    print('Connection established with slave {}'.format(camera_idx))


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
        socks = {'L': create_socket(context, L_PORT), 'R': create_socket(context, R_PORT)}
        print('Waiting on ports {} and {}...'.format(L_PORT, R_PORT))

        # Set up other environment variables
        img_folders = {'L': L_IMG_FOLDER, 'R': R_IMG_FOLDER}

        # Accept connections in a thread pool
        with ThreadPoolExecutor() as executor:
            executor.submit(accept_slave_thread, socks['L'], 'L')
            executor.submit(accept_slave_thread, socks['R'], 'R')

        # Confirm connection to both slaves by sending a message
        msg = 'Connection established with both slaves'
        print(msg)
        concurrent_send(socks, msg)

        # User input cycle
        while True:
            # Display action menu and ask for user input
            sel = user_input()

            # Tell slaves to prepare for streaming, by sending user's selection,
            # unless he chose calibration (done server-side only)
            if sel != 2:
                concurrent_send(socks, str(sel))
            if sel == 4:
                break

            # Invoke the corresponding function
            if sel == 1:
                capture_images(socks, img_folders, CAMERA_RESOLUTION)
            elif sel == 2:
                calibrate(img_folders, PATTERN_SIZE, SQUARE_LEN, CALIB_FILE)
            elif sel == 3:
                disp_map(socks, CALIB_FILE, CAMERA_RESOLUTION)
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
