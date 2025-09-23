import json
from multiprocessing import resource_tracker, shared_memory
import os
import socket
from typing import Any, Dict

import numpy as np
from my_ipc.public import ShmArrayInfo, ShmArray, generate_socket_path, generate_shm_name


class IPCServer:
    """IPC服务器基类"""
    
    def __init__(self, id: str):
        self.id = id
        self.socket_path = generate_socket_path(self.id)
        self.shm_arrs: Dict[str, ShmArray] = {}
    
    def start_server(self):
        """启动服务器监听"""
        server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        
        try:
            server_socket.bind(self.socket_path)
            server_socket.listen(1)
            
            client_socket, _ = server_socket.accept()

            data = client_socket.recv(4096).decode("utf-8")
            shm_infos = json.loads(data)
            for name, info_json in shm_infos.items():
                info = ShmArrayInfo.from_json(info_json)
                shm = shared_memory.SharedMemory(
                    name=generate_shm_name(self.id, name),
                    create=False,
                )
                # 对于 create = False 的共享内存，不要让 resource_tracker 去跟踪它, 否则会报警告
                resource_tracker.unregister(shm._name, "shared_memory") # type: ignore
                self.shm_arrs[name] = ShmArray(info=info, shm=shm)
            
            self.after_shm_created()
            
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
            for shm_arr in self.shm_arrs.values():
                shm_arr.shm.close()
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理请求的抽象方法，子类需要实现"""
        raise NotImplementedError
    
    def after_shm_created(self):
        """在共享内存创建后调用的钩子方法，子类可选实现"""
        pass
    
    def write_shared_array(self, data: np.ndarray, name: str = "default") -> None:
        """将numpy数组写入共享内存"""
        shm_arr = self.shm_arrs[name]
        assert data.shape == shm_arr.info.shape, f"Expected shape {shm_arr.info.shape}, but got {data.shape}"
        assert data.dtype == shm_arr.info.dtype, f"Expected dtype {shm_arr.info.dtype}, but got {data.dtype}"
        shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm_arr.shm.buf)
        np.copyto(shared_array, data)
    
    def get_shm_array_info(self, name: str = "default") -> ShmArrayInfo:
        """获取共享内存数组的信息"""
        return self.shm_arrs[name].info