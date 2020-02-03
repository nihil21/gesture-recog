import argparse
import zmq
from utils import calibration_tools as ct, network_tools as nt

# Ports of the sensors
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

# Resolution
RES = (416, 320)


def print_title():
    print('  ________                 __                                __________                            ')
    print(' /  _____/  ____   _______/  |_ __ _________   ____          \\______   \\ ____   ____  ____   ____  ')
    print('/   \\  ____/ __ \\ /  ___/\\   __\\  |  \\_  __ \\_/ __ \\   ______ |       _// __ \\_/ ___\\/  _ '
          '\\ / ___\\ ')
    print('\\    \\_\\  \\  ___/ \\___ \\  |  | |  |  /|  | \\/\\  ___/  /_____/ |    |   \\  ___/\\  \\__'
          '(  <_> ) /_/  >')
    print(' \\______  /\\___  >____  > |__| |____/ |__|    \\___  >         |____|_  /\\___  >\\___  >____/\\___  /')
    print('        \\/     \\/     \\/                          \\/                 \\/     \\/     \\/     /_____/  ')


def user_input() -> int:
    """Displays a menu of the available actions and asks user's input
        :return sel: integer representing user's choice"""

    print('')
    print('=' * 50)
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
    print('-' * 50)
    return sel


def main():
    # Construct argument parser and add arguments to it
    ap = argparse.ArgumentParser()
    ap.add_argument("-iL", "--ip_addressL", required=True, help="hostname of the left sensor")
    ap.add_argument("-iR", "--ip_addressR", required=True, help="hostname of the right sensor")
    args = vars(ap.parse_args())

    # Argument reading and check
    ipaddrL = args['ip_addressL']
    ipaddrR = args['ip_addressR']

    # Create dictionary of image folders paths
    img_folders = {'L': L_IMG_FOLDER, 'R': R_IMG_FOLDER}

    context = None
    socks = None
    try:
        # Set up context
        context = zmq.Context()
        # Create sockets and put them in a dictionary
        print('Trying to connect to sensors at {:s}:{:d} and {:s}:{:d}...'
              .format(ipaddrL, L_PORT, ipaddrR, R_PORT))
        socks = {'L': nt.create_socket_connect(context, ipaddrL, L_PORT),
                 'R': nt.create_socket_connect(context, ipaddrR, R_PORT)}
        # Wait for connection message from both sensors
        nt.concurrent_recv(socks)
        print('Connected to both sensors')
        nt.concurrent_send(socks, 'both connected')

        # Display the title of the tool in ASCII art
        print_title()

        while True:
            # Display action menu and ask for user input
            sel = user_input()
            # Send command to sensors, unless user chose to calibrate cameras (server-side only)
            if sel != 2:
                nt.concurrent_send(socks, str(sel))

            # Invoke corresponding function
            if sel == 1:
                ct.capture_images(socks, img_folders, RES)
            elif sel == 2:
                ct.calibrate_stereo_camera(img_folders, PATTERN_SIZE, SQUARE_LEN, CALIB_FILE, RES)
            elif sel == 3:
                ct.disp_map(socks, CALIB_FILE, RES)
            elif sel == 4:
                break
    except KeyboardInterrupt:
        print('\nTermination enforced manually')
    finally:
        # Close sockets and context
        for sock in socks.values():
            sock.close()
        context.term()
        print('Terminating...')


main()
