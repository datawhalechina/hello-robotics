"""Small NumPy-over-TCP protocol that keeps Isaac Sim and Motus in separate processes."""

from __future__ import annotations

from io import BytesIO
import socket
import struct

import numpy as np

_HEADER = struct.Struct("!Q")


def _read(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        block = sock.recv(size - len(data))
        if not block:
            raise ConnectionError("policy connection closed")
        data.extend(block)
    return bytes(data)


def send(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def receive(sock: socket.socket) -> bytes:
    return _read(sock, _HEADER.unpack(_read(sock, _HEADER.size))[0])


def pack(**values) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(buffer, **values)
    return buffer.getvalue()


def unpack(payload: bytes) -> dict:
    with np.load(BytesIO(payload), allow_pickle=False) as values:
        return {key: np.asarray(values[key]) for key in values.files}


class RemotePolicy:
    def __init__(self, host: str = "127.0.0.1", port: int = 8616, timeout: float = 900):
        self.address = (host, int(port))
        self.timeout = float(timeout)

    def predict(self, images, state: np.ndarray, instruction: str) -> np.ndarray:
        request = pack(
            head=np.asarray(images[0], np.uint8),
            left=np.asarray(images[1], np.uint8),
            right=np.asarray(images[2], np.uint8),
            state=np.asarray(state, np.float32),
            instruction=np.asarray(instruction),
        )
        with socket.create_connection(self.address, timeout=self.timeout) as connection:
            connection.settimeout(self.timeout)
            send(connection, request)
            answer = unpack(receive(connection))
        error = str(answer["error"].item())
        if error:
            raise RuntimeError(f"remote Motus error: {error}")
        return np.asarray(answer["actions"], np.float32)
