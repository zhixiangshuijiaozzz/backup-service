import logging
import socket
import threading
from typing import Dict, Optional

import nacos

logger = logging.getLogger("nacos")


def _guess_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return socket.gethostbyname(socket.gethostname())


class NacosClient:
    """轻量 Nacos 客户端，封装注册、心跳、下线"""

    def __init__(
        self,
        server_addr: str,
        service_name: str,
        ip: Optional[str],
        port: int,
        namespace: str = "",
        username: str | None = None,
        password: str | None = None,
        heartbeat_interval: int = 5,
        metadata: Optional[Dict[str, str]] = None,
        cluster_name: str = "DEFAULT",
        weight: float = 1.0,
    ):
        self.ip = ip or _guess_ip()
        self.port = int(port)
        self.service_name = service_name
        self.cluster_name = cluster_name
        self.weight = weight
        self.heartbeat_interval = max(2, int(heartbeat_interval))
        self.metadata = metadata or {}

        self._client = nacos.NacosClient(
            server_addresses=server_addr,
            namespace=namespace or None,
            username=username or None,
            password=password or None,
        )

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def register(self) -> bool:
        try:
            ok = self._client.register_instance(
                self.service_name,
                self.ip,
                self.port,
                cluster_name=self.cluster_name,
                metadata=self.metadata,
                weight=self.weight,
            )
            if ok:
                logger.info(f"[Nacos] 注册成功: {self.service_name}@{self.ip}:{self.port}")
                return True
            logger.error(f"[Nacos] 注册失败: {ok}")
        except Exception as exc:
            logger.error(f"[Nacos] 注册异常: {exc}")
        return False

    def heartbeat_once(self) -> bool:
        try:
            resp = self._client.send_heartbeat(
                self.service_name,
                self.ip,
                self.port,
                cluster_name=self.cluster_name,
                weight=self.weight,
                metadata=self.metadata,
            )
            if resp.get("clientBeatInterval"):
                logger.debug("[Nacos] 心跳 OK")
                return True
            logger.warning(f"[Nacos] 心跳异常: {resp}")
        except Exception as exc:
            logger.error(f"[Nacos] 心跳异常: {exc}")
        return False

    def deregister(self) -> bool:
        try:
            ok = self._client.deregister_instance(
                self.service_name,
                self.ip,
                self.port,
                cluster_name=self.cluster_name,
            )
            if ok:
                logger.info(f"[Nacos] 下线成功: {self.service_name}@{self.ip}:{self.port}")
                return True
            logger.warning(f"[Nacos] 下线失败: {ok}")
        except Exception as exc:
            logger.error(f"[Nacos] 下线异常: {exc}")
        return False

    def start(self):
        if not self.register():
            logger.warning("[Nacos] 首次注册失败，将继续尝试心跳与重试")
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            self.deregister()
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def _heartbeat_loop(self):
        interval = self.heartbeat_interval
        while not self._stop.is_set():
            ok = self.heartbeat_once()
            if not ok:
                self.register()
            self._stop.wait(interval)
