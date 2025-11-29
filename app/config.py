import os
import json
import requests
import logging

logger = logging.getLogger("config")

class Config:
    def __init__(self):
        self.config = {}
        self._load_local()
        self._load_remote_github()

    def _load_local(self):
        c = {}
        # —— 阿里云
        c["ALIYUN_ACCESS_KEY_ID"]     = os.getenv("ALIYUN_ACCESS_KEY_ID", "")
        c["ALIYUN_ACCESS_KEY_SECRET"] = os.getenv("ALIYUN_ACCESS_KEY_SECRET", "")
        c["ALIYUN_APP_KEY"]           = os.getenv("ALIYUN_APP_KEY", "")
        c["ALIYUN_REGION_ID"]         = os.getenv("ALIYUN_REGION_ID", "cn-shanghai")
        c["ALIYUN_DOMAIN"]            = os.getenv("ALIYUN_DOMAIN", "filetrans.cn-shanghai.aliyuncs.com")
        c["ALIYUN_API_VERSION"]       = os.getenv("ALIYUN_API_VERSION", "2018-08-17")

        # —— 文件服务
        c["FILE_UPLOAD_URL"]       = os.getenv("FILE_UPLOAD_URL", "https://www.tobeasy.cn/api/v1/files/upload")
        c["FILE_MULTI_UPLOAD_URL"] = os.getenv("FILE_MULTI_UPLOAD_URL", "https://www.tobeasy.cn/api/v1/files/upload/multi")
        c["FILE_DELETE_URL"]       = os.getenv("FILE_DELETE_URL", "https://www.tobeasy.cn/api/v1/files/byUrl")

        # —— RabbitMQ
        c["RABBITMQ_HOST"]     = os.getenv("RABBITMQ_HOST", "localhost")
        c["RABBITMQ_PORT"]     = int(os.getenv("RABBITMQ_PORT", "5672"))
        c["RABBITMQ_USER"]     = os.getenv("RABBITMQ_USER", "guest")
        c["RABBITMQ_PASS"]     = os.getenv("RABBITMQ_PASS", "guest")
        c["RABBITMQ_VHOST"]    = os.getenv("RABBITMQ_VHOST", "/")

        c["PYTHON_REQUEST_QUEUE"]        = os.getenv("PYTHON_REQUEST_QUEUE", "python.transcription.request.queue")
        c["PYTHON_REQUEST_EXCHANGE"]     = os.getenv("PYTHON_REQUEST_EXCHANGE", "python.request.exchange")
        c["PYTHON_REQUEST_ROUTING_KEY"]  = os.getenv("PYTHON_REQUEST_ROUTING_KEY", "python.transcription.request")

        c["TRANSCRIPTION_RESULT_QUEUE"]       = os.getenv("TRANSCRIPTION_RESULT_QUEUE", "transcription.result.queue")
        c["TRANSCRIPTION_RESULT_EXCHANGE"]    = os.getenv("TRANSCRIPTION_RESULT_EXCHANGE", "transcription.result.exchange")
        c["TRANSCRIPTION_RESULT_ROUTING_KEY"] = os.getenv("TRANSCRIPTION_RESULT_ROUTING_KEY", "transcription.result")

        # —— Redis
        c["REDIS_HOST"]     = os.getenv("REDIS_HOST", "localhost")
        c["REDIS_PORT"]     = int(os.getenv("REDIS_PORT", "6379"))
        c["REDIS_PASSWORD"] = os.getenv("REDIS_PASSWORD", "")
        c["REDIS_DB"]       = int(os.getenv("REDIS_DB", "0"))
        c["REDIS_TASK_KEY_PREFIX"] = os.getenv("REDIS_TASK_KEY_PREFIX", "task:")
        c["REDIS_TASK_TTL"] = int(os.getenv("REDIS_TASK_TTL", "3600"))

        # —— MySQL 日志
        c["MYSQL_HOST"]     = os.getenv("MYSQL_HOST", "8.133.197.6")
        c["MYSQL_PORT"]     = int(os.getenv("MYSQL_PORT", "3306"))
        c["MYSQL_DB"]       = os.getenv("MYSQL_DB", "tobeasy")
        c["MYSQL_USER"]     = os.getenv("MYSQL_USER", "root")
        c["MYSQL_PASSWORD"] = os.getenv("MYSQL_PASSWORD", "Lc146137")

        # —— HLS
        c["HLS_SEGMENT_TIME"] = int(os.getenv("HLS_SEGMENT_TIME", "6"))

        # —— 轮询配置
        c["MAX_POLLING_TIME"]  = int(os.getenv("MAX_POLLING_TIME", "300"))
        c["POLLING_INTERVAL"]  = int(os.getenv("POLLING_INTERVAL", "10"))
        c["HEARTBEAT_INTERVAL"] = int(os.getenv("HEARTBEAT_INTERVAL", "30"))

        # —— GitHub 远程配置地址
        c["CONFIG_GITHUB_URL"] = os.getenv("CONFIG_GITHUB_URL", "")

        c["EUREKA_ENABLED"] = os.getenv("EUREKA_ENABLED", "false").lower() in ("1", "true", "yes", "on")
        c["EUREKA_SERVER"] = os.getenv("EUREKA_SERVER", "")  # 例如 http://localhost:8761/eureka 或 http://localhost:8761
        c["EUREKA_APP_NAME"] = os.getenv("EUREKA_APP_NAME", "transcribe-service")
        c["EUREKA_PREFER_IP"] = os.getenv("EUREKA_PREFER_IP", "true").lower() in ("1", "true", "yes", "on")
        c["EUREKA_HEARTBEAT_INTERVAL"] = int(os.getenv("EUREKA_HEARTBEAT_INTERVAL", "30"))
        c["EUREKA_LEASE_RENEWAL_INTERVAL"] = int(os.getenv("EUREKA_LEASE_RENEWAL_INTERVAL", "30"))
        c["EUREKA_LEASE_EXPIRATION_DURATION"] = int(os.getenv("EUREKA_LEASE_EXPIRATION_DURATION", "90"))
        c["EUREKA_REGION"] = os.getenv("EUREKA_REGION", "default")
        c["EUREKA_ZONE"] = os.getenv("EUREKA_ZONE", "default")
        # 可选固定 host/ip（一般不需要，自动探测即可）
        c["EUREKA_INSTANCE_HOSTNAME"] = os.getenv("EUREKA_INSTANCE_HOSTNAME", "")
        c["EUREKA_INSTANCE_IP"] = os.getenv("EUREKA_INSTANCE_IP", "")

        self.config.update(c)

    def _load_remote_github(self):
        url = self.config.get("CONFIG_GITHUB_URL")
        if not url:
            logger.warning("未配置 CONFIG_GITHUB_URL，跳过远程配置加载")
            return
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                remote = json.loads(r.text)
                # 远端配置覆盖本地
                self.config.update(remote or {})
                logger.info("远程配置加载成功")
            else:
                logger.error(f"远程配置加载失败 code={r.status_code}")
        except Exception as e:
            logger.error(f"远程配置异常: {e}")

    def get(self, key, default=None):
        return self.config.get(key, default)

config = Config()
