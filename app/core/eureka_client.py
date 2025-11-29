# app/utils/eureka_client.py
import json
import logging
import socket
import threading
import time
from typing import Optional, Dict

import requests

logger = logging.getLogger("eureka")

class EurekaClient:
    """
    轻量 Eureka 客户端（基于 REST API）
    - 启动时注册：POST /eureka/apps/{APP_NAME}
    - 定时心跳：   PUT  /eureka/apps/{APP_NAME}/{INSTANCE_ID}
    - 优雅下线：   DELETE /eureka/apps/{APP_NAME}/{INSTANCE_ID}
    """

    def __init__(
        self,
        server: str,
        app_name: str,
        port: int,
        health_check_url: str,
        status_page_url: str,
        home_page_url: str,
        ip: Optional[str] = None,
        hostname: Optional[str] = None,
        prefer_ip: bool = True,
        heartbeat_interval: int = 30,
        lease_renewal_interval_in_seconds: int = 30,
        lease_expiration_duration_in_seconds: int = 90,
        metadata: Optional[Dict[str, str]] = None,
        region: str = "default",
        zone: str = "default",
        secure_port_enabled: bool = False,
        secure_port: Optional[int] = None,
    ):
        # 兼容 eureka server 地址格式： http://host:8761/eureka 或 http://host:8761
        self.server = server.rstrip("/")
        if not self.server.endswith("/eureka"):
            self.server += "/eureka"

        self.app_name = app_name.upper()
        self.port = int(port)
        self.ip = ip or self._guess_ip()
        self.hostname = hostname or self.ip
        self.prefer_ip = prefer_ip

        self.heartbeat_interval = int(heartbeat_interval)
        self.lease_renewal_interval_in_seconds = int(lease_renewal_interval_in_seconds)
        self.lease_expiration_duration_in_seconds = int(lease_expiration_duration_in_seconds)

        self.health_check_url = health_check_url
        self.status_page_url = status_page_url
        self.home_page_url = home_page_url

        self.secure_port_enabled = secure_port_enabled
        self.secure_port = int(secure_port) if secure_port_enabled and secure_port else None

        self.metadata = metadata or {}
        self.region = region
        self.zone = zone

        self.instance_id = f"{self.ip}:{self.app_name}:{self.port}"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _guess_ip() -> str:
        try:
            # 在容器或服务器上更可靠的取法
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return socket.gethostbyname(socket.gethostname())

    def _instance_payload(self) -> dict:
        payload = {
            "instance": {
                "instanceId": self.instance_id,
                "hostName": self.hostname,
                "app": self.app_name,
                "ipAddr": self.ip,
                "status": "UP",
                "overriddenstatus": "UNKNOWN",
                "port": {"$": self.port, "@enabled": "true"},
                "securePort": {"$": self.secure_port or 443, "@enabled": "true" if self.secure_port_enabled else "false"},
                "healthCheckUrl": self.health_check_url,
                "statusPageUrl": self.status_page_url,
                "homePageUrl": self.home_page_url,
                "vipAddress": self.app_name,
                "secureVipAddress": self.app_name,
                "countryId": 1,
                "dataCenterInfo": {
                    "@class": "com.netflix.appinfo.InstanceInfo$DefaultDataCenterInfo",
                    "name": "MyOwn"
                },
                "leaseInfo": {
                    "renewalIntervalInSecs": self.lease_renewal_interval_in_seconds,
                    "durationInSecs": self.lease_expiration_duration_in_seconds
                },
                "metadata": {
                    **self.metadata,
                    "management.port": str(self.port),
                    "zone": self.zone,
                    "region": self.region,
                }
            }
        }
        # preferIpAddress 让 Spring 客户端优先使用 IP
        if self.prefer_ip:
            payload["instance"]["metadata"]["preferIpAddress"] = "true"
        return payload

    # ------------------- 注册 / 心跳 / 下线 -------------------

    def register(self) -> bool:
        url = f"{self.server}/apps/{self.app_name}"
        try:
            resp = requests.post(url, data=json.dumps(self._instance_payload()), headers={"Content-Type": "application/json"}, timeout=5)
            if resp.status_code in (204, 200):  # 204=注册成功，200=已存在时覆盖
                logger.info(f"[Eureka] 注册成功: {self.instance_id}")
                return True
            logger.error(f"[Eureka] 注册失败 code={resp.status_code} body={resp.text}")
        except Exception as e:
            logger.error(f"[Eureka] 注册异常: {e}")
        return False

    def heartbeat_once(self) -> bool:
        url = f"{self.server}/apps/{self.app_name}/{self.instance_id}"
        try:
            resp = requests.put(url, timeout=5)
            if resp.status_code in (200, 204):
                logger.debug("[Eureka] 心跳 OK")
                return True
            logger.warning(f"[Eureka] 心跳失败 code={resp.status_code} body={resp.text}")
        except Exception as e:
            logger.error(f"[Eureka] 心跳异常: {e}")
        return False

    def deregister(self) -> bool:
        url = f"{self.server}/apps/{self.app_name}/{self.instance_id}"
        try:
            resp = requests.delete(url, timeout=5)
            if resp.status_code in (200, 202, 204):
                logger.info(f"[Eureka] 下线成功: {self.instance_id}")
                return True
            logger.warning(f"[Eureka] 下线失败 code={resp.status_code} body={resp.text}")
        except Exception as e:
            logger.error(f"[Eureka] 下线异常: {e}")
        return False

    # ------------------- 生命周期控制 -------------------

    def start(self):
        # 先注册一次
        if not self.register():
            logger.warning("[Eureka] 首次注册失败，将继续尝试心跳与重试")
        # 开启心跳线程
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
        """心跳 + 自愈注册"""
        interval = max(5, self.heartbeat_interval)
        while not self._stop.is_set():
            ok = self.heartbeat_once()
            if not ok:
                # 心跳失败则尝试重新注册（Eureka 重启等场景）
                self.register()
            self._stop.wait(interval)
