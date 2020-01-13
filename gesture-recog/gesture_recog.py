import cv2
import numpy as np
import zmq
import base64
from concurrent.futures import ThreadPoolExecutor

# Ports for both cameras
DX_PORT = 8000
SX_PORT = 8001


def create_socket(context: zmq.Context, tcp_port: int) -> zmq.Socket:
    sock = context.socket(zmq.PAIR)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind('tcp://*:{:d}'.format(tcp_port))

    return sock


def accept_clients(sock: zmq.Socket, sock_idx: str):
    msg = '{} client connected'.format(sock_idx)

    # Send message to client
    sock.send_string(msg)
    print(msg)


def receive_frame_thread(sock: zmq.Socket, sock_idx: str):
    # Send message to client
    sock.send_string('Both clients connected, transmission can start')

    while True:
        # Read stream in buffers of 4096 until a termination character '\0' is found
        serial_frame = sock.recv_string()
        frame = cv2.imdecode(np.fromstring(base64.b64decode(serial_frame), dtype=np.uint8), 1)

        cv2.imshow('{} frame'.format(sock_idx), frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # If 'q' is pressed, the termination signal is sent to the client
            sock.send_string('\0')
            cv2.destroyAllWindows()
            break


def main():
    try:
        # Set up zmq context and sockets PAIR
        context = zmq.Context()
        socks = {'DX': create_socket(context, DX_PORT), 'SX': create_socket(context, SX_PORT)}
        print('Waiting on ports {} and {}...'.format(DX_PORT, SX_PORT))

        # Accept connections in a thread pool
        with ThreadPoolExecutor() as executor:
            executor.submit(accept_clients, socks['DX'], 'DX')
            #executor.submit(accept_clients, socks['SX'], 'SX')

        print('Both clients connected, transmission can start.')

        # Receive frames by both clients using threads
        with ThreadPoolExecutor() as executor:
            executor.submit(receive_frame_thread, socks['DX'], 'DX')
            #executor.submit(receive_frame_thread, socks['SX'], 'SX')

    finally:
        # Closing sockets
        for sock in socks.values():
            sock.close()
        context.term()
        print('Terminating...')


main()
