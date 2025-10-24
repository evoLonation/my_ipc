from dataclasses import dataclass
from multiprocessing import resource_tracker
import enum
import json
from multiprocessing import shared_memory
import socket
from typing import Tuple

import numpy as np


class IPCMessageType(enum.Enum):
    QUIT = "QUIT"
    ERROR = "ERROR"
    TMP_SHARED_ARRAY = "TMP_SHARED_ARRAY"


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


class ShmArray:
    def __init__(self, info: ShmArrayInfo, name: str, create: bool = False):
        self.info = info
        self.need_unlink = create
        if create:
            self.shm = shared_memory.SharedMemory(
                create=True,
                size=np.zeros(info.shape, dtype=info.dtype).nbytes,
                name=name,
            )
        else:
            self.shm = shared_memory.SharedMemory(name=name, create=False)
            # 对于 create = False 的共享内存，不要让 resource_tracker 去跟踪它, 否则会报警告
            resource_tracker.unregister(self.shm._name, "shared_memory")  # type: ignore

    def get_info(self) -> ShmArrayInfo:
        return self.info

    def write(self, data: np.ndarray):
        assert (
            data.shape == self.info.shape
        ), f"Expected shape {self.info.shape}, but got {data.shape}"
        assert (
            data.dtype == self.info.dtype
        ), f"Expected dtype {self.info.dtype}, but got {data.dtype}"
        shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=self.shm.buf)
        np.copyto(shared_array, data)

    def read(self) -> np.ndarray:
        shared_array = np.ndarray(
            self.info.shape, dtype=self.info.dtype, buffer=self.shm.buf
        )
        result = np.zeros(self.info.shape, dtype=self.info.dtype)
        np.copyto(result, shared_array)
        return result

    def close(self):
        self.shm.close()
        if self.need_unlink:
            self.shm.unlink()
            self.need_unlink = False  # 防止重复 unlink

    def __del__(self):
        self.close()


def generate_socket_path(id: str) -> str:
    return f"/tmp/ipc_socket_{id}"


def generate_shm_name(id: str, name: str) -> str:
    return f"ipc_shm_{id}_{name}"


def recv_exactly(sock: socket.socket, n: int) -> bytes:
    """接收确切的 n 个字节"""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("连接已关闭，数据不完整")
        data += chunk
    return data


def recv_str(sock: socket.socket) -> str:
    # 先接受一个数字作为 bufsize
    bufsize_bytes = sock.recv(4)
    bufsize = int.from_bytes(bufsize_bytes, byteorder="big")
    data = recv_exactly(sock, bufsize).decode("utf-8")
    return data


def send_str(sock: socket.socket, data: str):
    data_bytes = data.encode("utf-8")
    bufsize = len(data_bytes)
    bufsize_bytes = bufsize.to_bytes(4, byteorder="big")
    sock.sendall(bufsize_bytes)
    sock.sendall(data_bytes)
