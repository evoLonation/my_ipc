from multiprocessing import shared_memory
import subprocess as sp
import json
import os
import socket
import time
from typing import Any, Dict, Iterable, Union, cast, overload, Tuple
import uuid

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


class IPCClient:

    def __init__(
        self,
        server_cmd: str,  # 启动服务器的命令，需包含 {id} 占位符
        shm_arrs: Union[Dict[str, ShmArrayInfo], ShmArrayInfo] = {},
        max_wait: int = 60,
    ):
        self.id = uuid.uuid4().hex
        self.socket_path = generate_socket_path(self.id)
        if isinstance(shm_arrs, ShmArrayInfo):
            shm_arrs = {"default": shm_arrs}
        self.shm_arrs: Dict[str, ShmArray] = {
            name: ShmArray(
                info=shm_arr, name=generate_shm_name(self.id, name), create=True
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
            self.process.terminate()
            raise RuntimeError("服务器启动超时")

        # 连接socket
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.connect(self.socket_path)

        shm_infos = {
            name: shm_arr.get_info().to_json()
            for name, shm_arr in self.shm_arrs.items()
        }
        send_str(self.socket, json.dumps(shm_infos))

    @overload
    def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]: ...

    @overload
    def send_request(
        self, request: Dict[str, Any], tmp_shm: ShmArrayInfo
    ) -> Tuple[Dict[str, Any], np.ndarray]: ...

    def send_request(
        self, request: Dict[str, Any], tmp_shm: Union[ShmArrayInfo, None] = None
    ):
        tmp_shm_arr = None
        if tmp_shm is not None:
            tmp_name = uuid.uuid4().hex
            tmp_shm_arr = ShmArray(
                info=tmp_shm,
                name=generate_shm_name(self.id, tmp_name),
                create=True,
            )
            send_str(self.socket, IPCMessageType.TMP_SHARED_ARRAY.value)
            send_str(
                self.socket,
                json.dumps(
                    {
                        "info": tmp_shm.to_json(),
                        "name": tmp_name,
                    }
                ),
            )

        send_str(self.socket, json.dumps(request))
        response_data = recv_str(self.socket)
        if response_data == "ERROR":
            raise RuntimeError("服务器处理请求时出错")
        if tmp_shm is not None:
            tmp_shm_arr = cast(ShmArray, tmp_shm_arr)
            tmp_data = tmp_shm_arr.read()
            tmp_shm_arr.close()
            return json.loads(response_data), tmp_data
        else:
            return json.loads(response_data)

    def send_stream_request(
        self, request: Dict[str, Any]
    ) -> Iterable[Dict[str, Any]]:
        """发送流式请求，返回一个迭代器，每次迭代返回一个响应"""
        send_str(self.socket, IPCMessageType.STREAM_REQUEST.value)
        send_str(self.socket, json.dumps(request))
        
        while True:
            response_type = recv_str(self.socket)
            if response_type == IPCMessageType.STREAM_END.value:
                break
            elif response_type == IPCMessageType.ERROR.value:
                raise RuntimeError("服务器处理流式请求时出错")
            elif response_type == IPCMessageType.STREAM_DATA.value:
                response_data = recv_str(self.socket)
                yield json.loads(response_data)
            else:
                raise RuntimeError(f"未知的响应类型: {response_type}")

    def get_shared_array(self, name: str = "default") -> ShmArray:
        return self.shm_arrs[name]

    def close(self):
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
                shm_arr.close()

    def __del__(self):
        self.close()
