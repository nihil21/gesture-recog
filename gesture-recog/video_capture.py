import sys
import argparse
import zmq
from model.network_agent import ImageSender
from typing import Tuple

# Ports of the sensors
L_PORT: int = 8000
R_PORT: int = 8001

# Camera size
RES: Tuple[int, int] = (640, 480)


def main():
    # Construct argument parser and add arguments to it
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--orientation', required=True, help="left/right orientation of the sensor ('L'/'R')")
    ap.add_argument('-r', '--rotate', action='store_true', help='if set, image is rotated by 180 degrees before being '
                                                                'sent to master (useful in particular hardware setups)')
    args = vars(ap.parse_args())

    # Argument reading and check
    orientation = args['orientation']
    if orientation == 'L':
        port = L_PORT
    elif orientation == 'R':
        port = R_PORT
    else:
        sys.exit("Argument 'orientation' must be either 'L' for left or 'R' for right.")
    rotate = args['rotate']

    # Create Sensor object
    sender = ImageSender(port, RES, rotate)

    try:
        while True:
            print(f'Waiting on port {port}...')
            sig = sender.recv_msg()
            print(f'Master: {sig}')

            # Start streaming
            sender.stream()
    except zmq.ZMQError:
        print('\nError communicating over the network')
    except KeyboardInterrupt:
        print('\nTermination enforced manually')
    finally:
        # Free resources
        sender.close()
        print('Terminating...')


if __name__ == '__main__':
    main()
