# 参考 /mnt/ssd/home/zhaozy/my_ipc/src/my_ipc/ipc_server.py 的实现
# 去掉了所有共享内存相关的逻辑，用于直接嵌入到其他 Python 文件中使用

import json
import os
import socket
from typing import Any, Dict
import enum


class IPCMessageType(enum.Enum):
    QUIT = "QUIT"
    ERROR = "ERROR"
    STREAM_REQUEST = "STREAM_REQUEST"
    STREAM_DATA = "STREAM_DATA"
    STREAM_END = "STREAM_END"


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


class IPCServer:
    """IPC服务器基类"""

    def __init__(self, id: str):
        self.id = id
        self.socket_path = generate_socket_path(self.id)

    def start_server(self):
        """启动服务器监听"""
        server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        try:
            server_socket.bind(self.socket_path)
            server_socket.listen(1)

            client_socket, _ = server_socket.accept()

            # 接收初始化信息（不包含共享内存信息）
            data = recv_str(client_socket)
            init_info = json.loads(data)

            self.after_init(init_info)

            while True:
                data = recv_str(client_socket)
                if not data or data.strip() == IPCMessageType.QUIT.value:
                    break

                if data.strip() == IPCMessageType.STREAM_REQUEST.value:
                    # 处理流式请求
                    request = json.loads(recv_str(client_socket))
                    try:
                        for response in self.handle_stream_request(request):
                            send_str(client_socket, IPCMessageType.STREAM_DATA.value)
                            send_str(client_socket, json.dumps(response))
                        send_str(client_socket, IPCMessageType.STREAM_END.value)
                    except Exception:
                        send_str(client_socket, IPCMessageType.ERROR.value)
                        raise
                else:
                    # 处理普通请求
                    request = json.loads(data)
                    try:
                        response = self.handle_request(request)
                    except Exception:
                        send_str(client_socket, IPCMessageType.ERROR.value)
                        raise
                    send_str(client_socket, json.dumps(response))

            client_socket.close()

        finally:
            server_socket.close()
            os.unlink(self.socket_path)

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理请求的抽象方法，子类需要实现
        """
        raise NotImplementedError

    def handle_stream_request(self, request: Dict[str, Any]):
        """
        处理流式请求的抽象方法，子类需要实现
        返回一个迭代器，每次迭代返回一个响应字典
        """
        raise NotImplementedError

    def after_init(self, init_info: Dict[str, Any]):
        """在初始化后调用的钩子方法，子类可选实现"""
        pass
