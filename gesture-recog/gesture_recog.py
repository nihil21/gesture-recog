import argparse
import zmq
from model.stereo_camera import StereoCamera
from model.errors import *
from typing import Tuple

# Ports of the sensors
L_PORT: int = 8000
R_PORT: int = 8001

# Folders to store images for calibration
IMG_FOLDER: str = "../calibration-images/"

# Chessboard size
PATTERN_SIZE: Tuple[int, int] = (8, 5)

# Square length
SQUARE_LEN: float = 26.5  # mm

# File where calibration data is stored
CALIB_FILE: str = "../calibration-data/calib"  # .npz

# File where disparity data is stored
DISP_FILE: str = "../calibration-data/disp"  # .npz


def print_title():
    print('  ________                 __                                __________                            ')
    print(' /  _____/  ____   _______/  |_ __ _________   ____          \\______   \\ ____   ____  ____   ____  ')
    print('/   \\  ____/ __ \\ /  ___/\\   __\\  |  \\_  __ \\_/ __ \\   ______ |       _// __ \\_/ ___\\/  _ '
          '\\ / ___\\ ')
    print('\\    \\_\\  \\  ___/ \\___ \\  |  | |  |  /|  | \\/\\  ___/  /_____/ |    |   \\  ___/\\  \\__'
          '(  <_> ) /_/  >')
    print(' \\______  /\\___  >____  > |__| |____/ |__|    \\___  >         |____|_  /\\___  >\\___  >____/\\___  /')
    print('        \\/     \\/     \\/                          \\/                 \\/     \\/     \\/     /_____/  ')
    print('-' * 99)


def user_input() -> int:
    """Displays a menu of the available actions and asks user's input
        :return sel: integer representing user's choice"""

    print('')
    print('=' * 50)
    print('1 - Collect images for calibration')
    print('2 - Perform calibration')
    print('3 - Disparity map tuning on sample images')
    print('4 - Real time disparity map')
    print('5 - Exit')

    while True:
        try:
            sel = input('Select one of the options [1, 2, 3, 4, 5]: ')
            sel = int(sel)
            if sel not in [1, 2, 3, 4, 5]:
                print('The option inserted is not valid, retry.')
            else:
                break
        except ValueError:
            print('The option inserted is not numeric, retry.')
    print('-' * 50)
    return sel


def main():
    # Display the title of the tool in ASCII art
    print_title()

    # Construct argument parser and add arguments to it
    ap = argparse.ArgumentParser()
    ap.add_argument('-iL', '--ip_addrL', required=True, help='hostname of the left sensor')
    ap.add_argument('-iR', '--ip_addrR', required=True, help='hostname of the right sensor')
    args = vars(ap.parse_args())

    # Create two dictionaries for left and right endpoints
    hostL = dict(ip_addr=args['ip_addrL'],
                 port=L_PORT)
    hostR = dict(ip_addr=args['ip_addrR'],
                 port=R_PORT)

    # Create StereoCamera object
    print('Local endpoints towards sensors at {0:s}:{1:d} and {2:s}:{3:d} created'
          .format(hostL['ip_addr'], hostL['port'], hostR['ip_addr'], hostR['port']))
    stereo_camera = StereoCamera(hostL, hostR)
    try:
        stereo_camera.load_calib_params(CALIB_FILE)
        print('Calibration parameters loaded from file {0:s}, stereo camera is already calibrated'.format(CALIB_FILE))
    except IOError:
        print('Could not load calibration parameters from file {0:s}, stereo camera must be calibrated before usage'
              .format(CALIB_FILE))
    try:
        stereo_camera.load_disp_params(DISP_FILE)
        print('Disparity parameters loaded from file {0:s}, stereo camera has already the optimal parameters'
              .format(DISP_FILE))
    except IOError:
        print('Could not load disparity parameters from file {0:s}, optimal parameters must be tuned before usage'
              .format(DISP_FILE))

    try:
        while True:
            # Display action menu and ask for user input
            sel = user_input()

            # Invoke corresponding function
            if sel == 1:
                stereo_camera.multicast_send('connected')
                stereo_camera.capture_sample_images(IMG_FOLDER)
            elif sel == 2:
                try:
                    stereo_camera.calibrate(IMG_FOLDER, PATTERN_SIZE, SQUARE_LEN, CALIB_FILE)
                except CalibrationImagesNotFoundError as e:
                    print('There are no images to perform calibration in folder {0:s}, collect them first'
                          .format(e.folder))
                except ChessboardNotFoundError as e:
                    print('No chessboards were detected in the images provided in folder {0:s}, capture better images'
                          .format(e.file))
            elif sel == 3:
                stereo_camera.multicast_send('connected')
                try:
                    stereo_camera.disp_map_tuning(DISP_FILE)
                except MissingParametersError as e:
                    print('{0:s} parameters missing'.format(e.parameter_cat))
            elif sel == 4:
                stereo_camera.multicast_send('connected')
                stereo_camera.realtime_disp_map()
            elif sel == 5:
                break
    except zmq.ZMQError:
        print('\nError communicating over the network')
    except KeyboardInterrupt:
        print('\nTermination enforced manually')
    finally:
        # Free resources
        stereo_camera.close()
        print('Terminating...')


if __name__ == '__main__':
    main()
