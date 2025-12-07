# app/main.py
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from app.api import chatgpt, health, pdf_translation
from app.core.logging_utils import setup_logging
from app.config import config
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import status, Request
from app.core.nacos_client import NacosClient, _guess_ip
import logging
import subprocess

setup_logging()
app = FastAPI(title="Transcribe Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 生产按需收紧
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chatgpt.router, prefix="/api")
app.include_router(health.router, prefix="/api")

app.include_router(pdf_translation.router, prefix="/api")
_nacos: NacosClient | None = None

@app.on_event("startup")
async def on_startup():
    logging.info("媒体服务启动")
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        logging.info("ffmpeg 可用")
    except Exception as e:
        logging.error(f"依赖检查失败: {e}")

    port = 8091  # 如果用 uvicorn 外部传端口，这里可从环境变量读取

    # ===== Nacos 注册 =====
    if config.get("NACOS_ENABLED"):
        try:
            server_addr = config.get("NACOS_SERVER_ADDR")
            namespace = config.get("NACOS_NAMESPACE")
            username = config.get("NACOS_USERNAME") or None
            password = config.get("NACOS_PASSWORD") or None
            service_name = config.get("NACOS_SERVICE_NAME")
            heartbeat_interval = config.get("NACOS_HEARTBEAT_INTERVAL")

            instance_ip = _guess_ip()
            health_url = f"http://{instance_ip}:{port}/api/health"

            global _nacos
            _nacos = NacosClient(
                server_addr=server_addr,
                service_name=service_name,
                ip=instance_ip,
                port=port,
                namespace=namespace,
                username=username,
                password=password,
                heartbeat_interval=heartbeat_interval,
                metadata={"health_check": health_url},
            )
            _nacos.start()
        except Exception as e:
            logging.error(f"Nacos 初始化失败: {e}")

# =========================
# 异常处理
# =========================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = b""
    async for chunk in request.stream():
        body += chunk
    error_content = {"detail": jsonable_encoder(exc.errors())}
    try:
        error_content["body"] = body.decode("utf-8")
    except UnicodeDecodeError:
        preview_size = min(50, len(body))
        error_content["body_info"] = {
            "content_type": request.headers.get("content-type", "unknown"),
            "size_bytes": len(body),
            "note": "Binary data cannot be displayed as text",
            "hex_preview": body[:preview_size].hex()
        }
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=error_content)


@app.on_event("shutdown")
async def on_shutdown():
    logging.info("媒体服务关闭")
    # 优雅下线 Nacos
    try:
        if _nacos:
            _nacos.stop()
    except Exception:
        pass

