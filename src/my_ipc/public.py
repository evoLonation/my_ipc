from dataclasses import dataclass
import json
from multiprocessing import shared_memory
import socket
from typing import Tuple

import numpy as np


@dataclass
class ShmArrayInfo:
    shape: Tuple[int, ...]
    dtype: type

    def to_json(self) -> str:
        return json.dumps({"shape": self.shape, "dtype": np.dtype(self.dtype).str})

    @staticmethod
    def from_json(data: str) -> "ShmArrayInfo":
        obj = json.loads(data)
        return ShmArrayInfo(
            shape=tuple(obj["shape"]),
            dtype=np.dtype(obj["dtype"]).type,
        )


@dataclass
class ShmArray:
    info: ShmArrayInfo
    shm: shared_memory.SharedMemory


def generate_socket_path(id: str) -> str:
    return f"/tmp/ipc_socket_{id}"

def generate_shm_name(id: str, name: str) -> str:
    return f"ipc_shm_{id}_{name}"


def recv_str(sock: socket.socket) -> str:
    # 先接受一个数字作为 bufsize
    bufsize_bytes = sock.recv(4)
    bufsize = int.from_bytes(bufsize_bytes, byteorder="big")
    data = sock.recv(bufsize).decode("utf-8")
    return data

def send_str(sock: socket.socket, data: str):
    data_bytes = data.encode("utf-8")
    bufsize = len(data_bytes)
    bufsize_bytes = bufsize.to_bytes(4, byteorder="big")
    sock.sendall(bufsize_bytes)
    sock.sendall(data_bytes)