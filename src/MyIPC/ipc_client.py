from multiprocessing import shared_memory
import subprocess as sp
import json
import os
import socket
import time
from typing import Any
import uuid

import numpy as np


class IPCClient:
    """IPC客户端"""

    def __init__(
        self,
        server_cmd: str,  # 启动服务器的命令，需包含 {socket_path} 和 {shm_name} 占位符
        shm_shape: tuple[int, ...],
        shm_dtype: type = np.float32,
        max_wait: int = 60,
    ):
        self.id = uuid.uuid4().hex
        self.socket_path = f"/tmp/ipc_socket_{self.id}"
        self.shm_name = f"ipc_shm_{self.id}"
        self.shm_shape = shm_shape
        self.shm_dtype = shm_dtype

        # 创建共享内存
        self.shm = shared_memory.SharedMemory(
            create=True,
            size=np.zeros(shm_shape, dtype=shm_dtype).nbytes,
            name=self.shm_name,
        )

        # 启动服务器进程
        cmd = server_cmd.format(socket_path=self.socket_path, shm_name=self.shm_name)
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

    def send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """发送请求并接收响应"""
        self.socket.send(json.dumps(request).encode("utf-8"))
        response_data = self.socket.recv(4096).decode("utf-8")
        if response_data == "ERROR":
            raise RuntimeError("服务器处理请求时出错")
        return json.loads(response_data)

    def read_shared_array(self) -> np.ndarray:
        """从共享内存读取numpy数组"""
        shared_array = np.ndarray(
            self.shm_shape, dtype=self.shm_dtype, buffer=self.shm.buf
        )
        result = np.zeros(self.shm_shape, dtype=self.shm_dtype)
        np.copyto(result, shared_array)
        return result

    def close(self):
        """关闭连接和清理资源"""
        if hasattr(self, "socket"):
            try:
                self.socket.send("QUIT".encode("utf-8"))
            except Exception:
                pass
            self.socket.close()
        if hasattr(self, "process"):
            try:
                self.process.wait(timeout=5)
            except sp.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        if hasattr(self, "shm"):
            self.shm.close()
            self.shm.unlink()

    def __del__(self):
        self.close()
