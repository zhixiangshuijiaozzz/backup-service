# app/api/transcribe.py
from fastapi import APIRouter, Form, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import json, uuid, os, tempfile, logging, gc

from app.core.aliyun_filetrans import AliyunFileTrans
from app.core.mq_utils import (
    create_rabbitmq_connection,
    setup_rabbitmq_channel,
    send_result_to_exchange,
)
from app.core.video_utils import (
    upload_file_to_server,
    delete_file_by_url,
)
from app.config import config
from app.services.transcription_service import run_transcription_pipeline

router = APIRouter(tags=["Transcribe"])

logger = logging.getLogger("transcribe-api")


def format_duration(seconds: float) -> str:
    """保留一个工具，给 /transcribe 使用；HLS 流水线已经用 service 里的 format_duration。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


@router.post("/transcribe")
async def transcribe_audio(
    file_url: str = Form(...),
    task_id: str = Form(...)
):
    """
    场景一：前端 / Java 已经把音频上传好了，只传一个 file_url 过来。
    逻辑保持你原来的：调用阿里云 + MQ + 返回简要信息。
    """
    try:
        logging.info(f"转写请求: taskId={task_id}, file={file_url}")
        client = AliyunFileTrans(
            config.get("ALIYUN_ACCESS_KEY_ID"),
            config.get("ALIYUN_ACCESS_KEY_SECRET")
        )
        aliyun_task_id = client.submit_file_trans_request(
            config.get("ALIYUN_APP_KEY"),
            file_url
        )
        if not aliyun_task_id:
            raise HTTPException(status_code=500, detail="提交录音文件识别请求失败")

        result_json = client.get_file_trans_result(aliyun_task_id)
        if result_json is None:
            raise HTTPException(status_code=500, detail="识别结果查询失败")

        formatted = client.format_result_for_mq(result_json)

        # 可选：发送到 MQ
        try:
            conn = create_rabbitmq_connection()
            ch = setup_rabbitmq_channel(
                conn,
                config.get("TRANSCRIPTION_RESULT_EXCHANGE"),
                config.get("TRANSCRIPTION_RESULT_QUEUE"),
                config.get("TRANSCRIPTION_RESULT_ROUTING_KEY"),
                is_result_queue=True
            )
            response = {
                'taskId': task_id,
                'text': formatted.get('text', ''),
                'language': formatted.get('language', 'zh'),
                'segments': formatted.get('segments', []),
                'success': True,
                'errorMessage': None
            }
            send_result_to_exchange(
                ch,
                config.get("TRANSCRIPTION_RESULT_EXCHANGE"),
                config.get("TRANSCRIPTION_RESULT_ROUTING_KEY"),
                response
            )
            conn.close()
        except Exception as mq_e:
            logging.error(f"发送 MQ 失败: {mq_e}")

        return {
            'taskId': task_id,
            'text': formatted.get('text', ''),
            'language': formatted.get('language', 'zh'),
            'segments': formatted.get('segments', []),
            'success': True,
            'errorMessage': None
        }
    except Exception as e:
        logging.error(f"转写失败: {e}")
        return {
            'taskId': task_id,
            'text': '',
            'language': '',
            'segments': [],
            'success': False,
            'errorMessage': str(e)
        }


@router.post("/convert_and_transcribe")
async def convert_and_transcribe(
    video: UploadFile = File(...),
    cookie: str = Form(...),
):
    """
    场景二：直接上传视频 / 音频文件到 Python。
    现在改为统一调用 run_transcription_pipeline，
    同时把完整结果同步到 MQ，并返回给前端一个简化结构。
    """
    tmp_path: str | None = None
    task_id = str(uuid.uuid4())

    try:
        # 1. 保存临时文件
        tmp_dir = tempfile.gettempdir()
        # 尽量保留原后缀
        suffix = video.filename.rsplit(".", 1)[-1] if "." in video.filename else "mp4"
        tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.{suffix}")
        content = await video.read()
        with open(tmp_path, "wb") as f:
            f.write(content)
        logging.info(f"[convert_and_transcribe] 已保存临时媒体: {tmp_path}")

        # 2. 统一流水线：HLS + 阿里云转写 + 分组 + 图片
        pipeline_result = run_transcription_pipeline(
            tmp_path,
            cookie,
            language_hint="zh",
            extract_images=True,
        )

        # 3. 发送完整结果到 MQ（结构与 Worker 保持一致）
        conn = None
        try:
            conn = create_rabbitmq_connection()
            ch = setup_rabbitmq_channel(
                conn,
                config.get("TRANSCRIPTION_RESULT_EXCHANGE"),
                config.get("TRANSCRIPTION_RESULT_QUEUE"),
                config.get("TRANSCRIPTION_RESULT_ROUTING_KEY"),
                is_result_queue=True,
            )
            response = {
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
                # HTTP 上传场景下暂时不透传 userId/extra
                "userId": None,
                "extra": {},
            }
            send_result_to_exchange(
                ch,
                config.get("TRANSCRIPTION_RESULT_EXCHANGE"),
                config.get("TRANSCRIPTION_RESULT_ROUTING_KEY"),
                response,
            )
        except Exception as mq_e:
            logging.error(f"[convert_and_transcribe] 发送结果到 MQ 失败: {mq_e}")
        finally:
            try:
                if conn and conn.is_open:
                    conn.close()
            except Exception:
                pass

        # 4. 返回给 HTTP 调用方一个简化版本
        return JSONResponse(
            content={
                "task_id": task_id,
                "duration": pipeline_result["durationSeconds"],
                "is_video": True,
                "transcription_result": {
                    "taskId": task_id,
                    "text": pipeline_result["text"],
                    "language": pipeline_result["language"],
                    "segments": pipeline_result["segments"],
                    "success": True,
                    "errorMessage": None,
                    "videoUrl": pipeline_result["videoUrl"],
                    "duration": pipeline_result["duration"],
                    # 方便前端做后处理的话，也可以把 sentences / groups 带出去
                    "sentences": pipeline_result["sentences"],
                    "groups": pipeline_result["groups"],
                },
            }
        )
    except Exception as e:
        logging.error(f"[convert_and_transcribe] 异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 删除临时视频
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
                logging.info(f"[convert_and_transcribe] 已删除临时文件: {tmp_path}")
        except Exception:
            pass
        gc.collect()
