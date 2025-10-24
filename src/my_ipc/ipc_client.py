from multiprocessing import shared_memory
import subprocess as sp
import json
import os
import socket
import time
from typing import Any, Dict
import uuid

import numpy as np
from my_ipc.public import (
    ShmArrayInfo,
    ShmArray,
    generate_socket_path,
    generate_shm_name,
    recv_str,
    send_str,
)


class IPCClient:
    """IPC客户端"""

    def __init__(
        self,
        server_cmd: str,  # 启动服务器的命令，需包含 {id} 占位符
        shm_arrs: dict[str, ShmArrayInfo] | ShmArrayInfo = {},
        max_wait: int = 60,
    ):
        self.id = uuid.uuid4().hex
        self.socket_path = generate_socket_path(self.id)
        if isinstance(shm_arrs, ShmArrayInfo):
            shm_arrs = {"default": shm_arrs}
        self.shm_arrs: Dict[str, ShmArray] = {
            name: ShmArray(
                info=shm_arr,
                shm=shared_memory.SharedMemory(
                    create=True,
                    size=np.zeros(shm_arr.shape, dtype=shm_arr.dtype).nbytes,
                    name=generate_shm_name(self.id, name),
                ),
            )
            for name, shm_arr in shm_arrs.items()
        }

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

        shm_infos = {
            name: shm_arr.info.to_json() for name, shm_arr in self.shm_arrs.items()
        }
        send_str(self.socket, json.dumps(shm_infos))

    def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """发送请求并接收响应"""
        send_str(self.socket, json.dumps(request))
        response_data = recv_str(self.socket)
        if response_data == "ERROR":
            raise RuntimeError("服务器处理请求时出错")
        return json.loads(response_data)

    def read_shared_array(self, name: str = "default") -> np.ndarray:
        """从共享内存读取numpy数组"""
        shm_arr = self.shm_arrs[name]

        shared_array = np.ndarray(
            shm_arr.info.shape, dtype=shm_arr.info.dtype, buffer=shm_arr.shm.buf
        )
        result = np.zeros(shm_arr.info.shape, dtype=shm_arr.info.dtype)
        np.copyto(result, shared_array)
        return result

    def close(self):
        """关闭连接和清理资源"""
        if hasattr(self, "socket"):
            try:
                send_str(self.socket, "QUIT")
            except Exception:
                pass
            self.socket.close()
        if hasattr(self, "process"):
            try:
                self.process.wait(timeout=5)
            except sp.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        if hasattr(self, "shm_arrs"):
            for shm_arr in self.shm_arrs.values():
                shm_arr.shm.close()
                shm_arr.shm.unlink()

    def __del__(self):
        self.close()
