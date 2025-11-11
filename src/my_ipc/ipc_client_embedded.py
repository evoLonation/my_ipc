# 参考 /mnt/ssd/home/zhaozy/my_ipc/src/my_ipc/ipc_client.py 的实现
# 去掉了所有共享内存相关的逻辑，用于直接嵌入到其他 Python 文件中使用

import subprocess as sp
import json
import os
import socket
import time
from typing import Any, Dict
import uuid
import enum


class IPCMessageType(enum.Enum):
    QUIT = "QUIT"
    ERROR = "ERROR"


def generate_socket_path(id: str) -> str:
    return f"/tmp/ipc_socket_{id}"


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


class IPCClient:

    def __init__(
        self,
        server_cmd: str,  # 启动服务器的命令，需包含 {id} 占位符
        max_wait: int = 60,
    ):
        self.id = uuid.uuid4().hex
        self.socket_path = generate_socket_path(self.id)

        # 启动服务器进程
        cmd = server_cmd.format(id=self.id)
        self.process = sp.Popen(cmd, shell=True, executable="/bin/bash")

        # 等待服务器就绪
        wait_time = 0
        while wait_time < max_wait:
            if os.path.exists(self.socket_path):
                break
            if self.process.poll() is not None:
                raise RuntimeError("服务器进程意外退出")
            time.sleep(1)
            wait_time += 1

        if not os.path.exists(self.socket_path):
            raise RuntimeError("服务器启动超时")

        # 连接socket
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.connect(self.socket_path)

        # 发送初始化信息（不包含共享内存信息）
        send_str(self.socket, json.dumps({}))

    def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        send_str(self.socket, json.dumps(request))
        response_data = recv_str(self.socket)
        if response_data == IPCMessageType.ERROR.value:
            raise RuntimeError("服务器处理请求时出错")
        return json.loads(response_data)

    def close(self):
        if hasattr(self, "socket"):
            try:
                send_str(self.socket, IPCMessageType.QUIT.value)
            except Exception:
                pass
            self.socket.close()
        if hasattr(self, "process"):
            try:
                self.process.wait(timeout=5)
            except sp.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)

    def __del__(self):
        self.close()
