import sys
import argparse
import zmq
from model.network_agent import ImageSender
from typing import Tuple

# Ports of the sensors
L_PORT: int = 8000
R_PORT: int = 8001

# Camera size
RES: Tuple[int, int] = (304, 304)


def main():
    # Construct argument parser and add arguments to it
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--orientation', required=True, help="left/right orientation of the sensor ('L'/'R')")
    ap.add_argument('-f', '--flip', action='store_true', help='if set, image is flipped before it is sent to master '
                                                              '(useful only in particular hardware setups)')
    args = vars(ap.parse_args())

    # Argument reading and check
    orientation = args['orientation']
    if orientation == 'L':
        port = L_PORT
    elif orientation == 'R':
        port = R_PORT
    else:
        sys.exit("Argument 'orientation' must be either 'L' for left or 'R' for right.")
    flip = args['flip']

    # Create Sensor object
    sender = ImageSender(port, RES, flip)

    try:
        while True:
            print('Waiting on port {0:d}...'.format(port))
            sig = sender.recv_msg()
            print('Master: {0:s}'.format(sig))

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
