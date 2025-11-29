# app/workers/consumer.py
# app/workers/consumer.py
import json
import logging
import os
import tempfile
import time
from typing import Dict, Any

import pika
import requests

from app.config import config
from app.core.redis_utils import lock_task_in_redis
from app.core.mq_utils import (
    create_rabbitmq_connection,
    setup_rabbitmq_channel,
    send_result_to_exchange,
)
from app.core.video_utils import has_video_stream
from app.services.transcription_service import run_transcription_pipeline

logger = logging.getLogger("worker")

HEARTBEAT_INTERVAL = config.get("HEARTBEAT_INTERVAL", 30)


def check_ffmpeg() -> bool:
    import subprocess

    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except FileNotFoundError:
        logger.error("FFmpeg 未安装或不可用")
        return False


def download_media_to_temp(url: str, task_id: str) -> str:
    """
    下载媒体到本地临时文件。
    采用流式下载，避免一次性加载大文件到内存。
    """
    logger.info(f"开始下载媒体: {url}")
    r = requests.get(url, stream=True, timeout=60)
    if r.status_code != 200:
        raise Exception(f"下载媒体失败: {r.status_code} {r.text}")

    suffix = ".mp4"
    # 从 URL 中简单提取后缀
    splitted = url.split("?")[0].split("#")[0]
    if "." in splitted:
        ext = splitted.rsplit(".", 1)[-1]
        if len(ext) <= 4:
            suffix = "." + ext

    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"task_{task_id}{suffix}")
    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    logger.info(f"媒体已下载到临时文件: {tmp_path}")
    return tmp_path


def process_task(ch, method, properties, body):
    """
    核心消费函数：
    - 预期消息体 body JSON:
      {
        "taskId": "...",           # 与 Java 侧 recordId 对齐
        "mediaUrl": "...",         # 原始媒体 URL（兼容 videoUrl/audioUrl）
        "language": "zh",          # 可选
        "userId": "...",           # 可选
        "extra": {...}             # 可选扩展
      }
    - properties.headers 中应包含:
      {
        "cookie": "xxx"
      }
    """
    if not check_ffmpeg():
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    task_id = None
    tmp_media_path: str | None = None

    try:
        msg = json.loads(body.decode())
        task_id = msg.get("taskId")
        media_url_source = (
            msg.get("mediaUrl")
            or msg.get("videoUrl")
            or msg.get("audioUrl")
        )
        language_hint = msg.get("language") or "zh"
        user_id = msg.get("userId")
        extra = msg.get("extra") or {}

        headers = properties.headers if properties and properties.headers else {}
        cookie = headers.get("cookie", "")

        logger.info(
            f"消费任务: taskId={task_id}, mediaUrl={media_url_source}, userId={user_id}"
        )

        # === 幂等控制（Redis）===
        if not lock_task_in_redis(task_id):
            logger.info(f"任务 {task_id} 不存在或已被处理，跳过")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        if not media_url_source:
            raise Exception("消息缺少 mediaUrl/videoUrl/audioUrl")

        # 1. 下载媒体到本地
        tmp_media_path = download_media_to_temp(media_url_source, task_id)
        has_video = has_video_stream(tmp_media_path)
        logger.info(
            f"任务媒体类型: {'video' if has_video else 'audio'}, 路径={tmp_media_path}"
        )

        # 2. 调统一流水线
        pipeline_result = run_transcription_pipeline(
            tmp_media_path,
            cookie,
            language_hint=language_hint,
            extract_images=has_video,
        )

        # 3. 组织回 MQ 的结果结构（给 Java TranscriptionResult 用）
        result_payload: Dict[str, Any] = {
            "taskId": task_id,
            "success": True,
            "errorMessage": None,
            "text": pipeline_result["text"],
            "language": pipeline_result["language"],
            "segments": pipeline_result["segments"],
            "sentences": pipeline_result["sentences"],
            "groups": pipeline_result["groups"],
            "subtitles": pipeline_result["subtitles"],
            "videoUrl": pipeline_result["videoUrl"],
            "duration": pipeline_result["duration"],
            # 透传字段
            "userId": user_id,
            "extra": extra,
        }

        send_result_to_exchange(
            ch,
            config.get("TRANSCRIPTION_RESULT_EXCHANGE"),
            config.get("TRANSCRIPTION_RESULT_ROUTING_KEY"),
            result_payload,
            headers=headers,
        )

        ch.basic_ack(delivery_tag=method.delivery_tag)
        logger.info(f"任务完成: taskId={task_id}")

    except Exception as e:
        logger.error(f"处理任务异常: {e}")
        # 尽量把错误也回传给 Java
        try:
            headers = properties.headers if properties and properties.headers else {}
            result_payload = {
                "taskId": task_id,
                "success": False,
                "errorMessage": str(e),
                "text": "",
                "language": "",
                "segments": [],
                "sentences": [],
                "groups": [],
                "subtitles": "",
                "videoUrl": "",
                "duration": "",
                "userId": None,
                "extra": {},
            }
            send_result_to_exchange(
                ch,
                config.get("TRANSCRIPTION_RESULT_EXCHANGE"),
                config.get("TRANSCRIPTION_RESULT_ROUTING_KEY"),
                result_payload,
                headers=headers,
            )
        except Exception as send_err:
            logger.error(f"发送错误结果失败: {send_err}")
        finally:
            try:
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception:
                pass
    finally:
        # 收尾清理：删除本地媒体
        try:
            if tmp_media_path and os.path.exists(tmp_media_path):
                os.remove(tmp_media_path)
                logger.info(f"已删除临时媒体: {tmp_media_path}")
        except Exception:
            pass


def main():
    logger.info("启动 Python Transcription Worker")
    while True:
        conn = None
        try:
            conn = create_rabbitmq_connection()
            req_ch = setup_rabbitmq_channel(
                conn,
                config.get("PYTHON_REQUEST_EXCHANGE"),
                config.get("PYTHON_REQUEST_QUEUE"),
                config.get("PYTHON_REQUEST_ROUTING_KEY"),
            )
            # 确保结果队列存在
            setup_rabbitmq_channel(
                conn,
                config.get("TRANSCRIPTION_RESULT_EXCHANGE"),
                config.get("TRANSCRIPTION_RESULT_QUEUE"),
                config.get("TRANSCRIPTION_RESULT_ROUTING_KEY"),
                is_result_queue=True,
            )

            req_ch.basic_consume(
                queue=config.get("PYTHON_REQUEST_QUEUE"),
                on_message_callback=process_task,
                auto_ack=False,
            )
            logger.info(f"开始消费队列: {config.get('PYTHON_REQUEST_QUEUE')}")
            req_ch.start_consuming()
        except KeyboardInterrupt:
            logger.info("Worker 收到中断信号，准备退出")
            try:
                if conn and conn.is_open:
                    conn.close()
            except Exception:
                pass
            break
        except pika.exceptions.AMQPConnectionError:
            logger.warning("AMQP 连接错误，5 秒后重试...")
            time.sleep(5)
            continue
        except Exception as e:
            logger.error(f"Worker 未知错误: {e}")
            time.sleep(5)
            continue
        finally:
            try:
                if conn and conn.is_open:
                    conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
