import socket


def recv_until_term(sock: socket, bufsize: int) -> bytes:
    data = []
    while True:
        packet = sock.recv(bufsize)
        if packet[-4:] == b'\term':
            data.append(packet[:-4])
            break
        data.append(packet)

    return b''.join(data)
