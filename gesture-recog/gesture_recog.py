import socket
import cv2
import pickle
from concurrent.futures import ThreadPoolExecutor
from utils.network_utils import recv_until_term

# Ports for both cameras
DX_PORT = 8000
SX_PORT = 8001


def create_server_socket(tcp_port: int) -> socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', tcp_port))
    sock.listen(1)

    return sock


def accept_clients(server_sock: socket, sock_idx: str) -> socket:
    client_sock, client_addr = server_sock.accept()
    msg = '{} client connected: {}'.format(sock_idx, client_addr)
    print(msg)

    # Send message to client
    client_sock.send(msg.encode(encoding='utf-8'))

    return client_sock


def receive_frame_thread(sock: socket, sock_idx: str):
    # Send message to client
    sock.send('Both clients connected, transmission can start'.encode(encoding='utf-8'))

    while True:
        # Read stream in buffers of 4096 until a termination character '\0' is found
        serial_frame = recv_until_term(sock, bufsize=4096)

        frame = pickle.loads(serial_frame)

        cv2.imshow('DX', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sock.send('Stop'.encode(encoding='utf-8'))
            break
        sock.send('Next'.encode(encoding='utf-8'))


def main():
    # Set up server sockets
    server_socks = {'DX': create_server_socket(DX_PORT), 'SX': create_server_socket(SX_PORT)}
    print('Waiting on ports {} and {}...'.format(DX_PORT, SX_PORT))

    # Accept connections in a thread pool
    client_socks = {}
    with ThreadPoolExecutor() as executor:
        future_dx = executor.submit(accept_clients, server_socks['DX'], 'DX')
        #future_sx = executor.submit(accept_clients, server_socks['SX'], 'SX')
        client_socks['DX'] = future_dx.result()
        #client_socks['SX'] = future_sx.result()

    print('Both clients connected, transmission can start.')

    # Receive frames by both clients using threads
    with ThreadPoolExecutor() as executor:
        future_dx = executor.submit(receive_frame_thread, client_socks['DX'], 'DX')
        #future_sx = executor.submit(receive_frame_thread, client_socks['SX'], 'SX')
        #frame = future_dx.result()

    # Close sockets
    for sock in server_socks.values():
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()


main()
