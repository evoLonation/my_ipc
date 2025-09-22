import json
from multiprocessing import resource_tracker, shared_memory
import os
import socket
from typing import Any

import numpy as np


class IPCServer:
    """IPC服务器基类"""
    
    def __init__(self, socket_path: str, shm_name: str):
        self.socket_path = socket_path
        self.shm_name = shm_name
        self.shm = shared_memory.SharedMemory(name=shm_name, create=False)
        # 对于 create = False 的共享内存，不要让 resource_tracker 去跟踪它, 否则会报警告
        resource_tracker.unregister(self.shm._name, "shared_memory") # type: ignore
    
    def start_server(self):
        """启动服务器监听"""
        server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        
        try:
            server_socket.bind(self.socket_path)
            server_socket.listen(1)
            
            client_socket, _ = server_socket.accept()
            
            while True:
                data = client_socket.recv(4096).decode("utf-8")
                if not data or data.strip() == "QUIT":
                    break
                
                request = json.loads(data)
                try:
                    response = self.handle_request(request)
                except Exception:
                    response = "ERROR"
                    response_bytes = response.encode("utf-8")
                    client_socket.send(response_bytes)
                    raise
                
                response_bytes = json.dumps(response).encode("utf-8")
                client_socket.send(response_bytes)
            
            client_socket.close()
        
        finally:
            server_socket.close()
            os.unlink(self.socket_path)
            self.shm.close()
    
    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """处理请求的抽象方法，子类需要实现"""
        raise NotImplementedError
    
    def write_shared_array(self, data: np.ndarray):
        """将numpy数组写入共享内存"""
        shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=self.shm.buf)
        np.copyto(shared_array, data)