import json
from multiprocessing import resource_tracker, shared_memory
import os
import socket
from typing import Any, Dict, Iterable, Union

import numpy as np
from my_ipc.public import (
    IPCMessageType,
    ShmArrayInfo,
    ShmArray,
    generate_socket_path,
    generate_shm_name,
    recv_str,
    send_str,
)


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

            data = recv_str(client_socket)
            shm_infos = json.loads(data)
            for name, info_json in shm_infos.items():
                info = ShmArrayInfo.from_json(info_json)
                self.shm_arrs[name] = ShmArray(
                    info=info, name=generate_shm_name(self.id, name), create=False
                )

            self.after_shm_created()

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
                    if data.strip() == IPCMessageType.TMP_SHARED_ARRAY.value:
                        shm_json = json.loads(recv_str(client_socket))
                        tmp_shm_arr = ShmArray(
                            info=ShmArrayInfo.from_json(shm_json["info"]),
                            name=generate_shm_name(self.id, shm_json["name"]),
                            create=False,
                        )
                        request = json.loads(recv_str(client_socket))
                    else:
                        tmp_shm_arr = None
                        request = json.loads(data)
                    try:
                        response = self.handle_request(request, tmp_shm=tmp_shm_arr)
                    except Exception:
                        send_str(client_socket, IPCMessageType.ERROR.value)
                        raise
                    send_str(client_socket, json.dumps(response))

            client_socket.close()

        finally:
            server_socket.close()
            os.unlink(self.socket_path)
            for shm_arr in self.shm_arrs.values():
                shm_arr.close()

    def handle_request(
        self, request: Dict[str, Any], tmp_shm: Union[ShmArray, None]
    ) -> Dict[str, Any]:
        """
        处理请求的抽象方法，子类需要实现
        tmp_shm: 只可以写入，不可以读取
        """
        raise NotImplementedError

    def handle_stream_request(
        self, request: Dict[str, Any]
    ) -> Iterable[Dict[str, Any]]:
        """
        处理流式请求的抽象方法，子类需要实现
        """
        raise NotImplementedError

    def after_shm_created(self):
        """在共享内存创建后调用的钩子方法，子类可选实现"""
        pass

    def get_shared_array(self, name: str = "default") -> ShmArray:
        return self.shm_arrs[name]
