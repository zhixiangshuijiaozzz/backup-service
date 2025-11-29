# app/services/transcription_service.py
import logging
from typing import Any, Dict, List

from app.config import config
from app.core.aliyun_filetrans import AliyunFileTrans
from app.core.video_utils import (
    get_media_duration,
    format_duration,
    convert_video_to_hls,
    upload_hls_files,
    convert_video_to_audio,
    upload_file_to_server,
    delete_file_by_url,
    extract_group_best_frames,
)
from app.core.transcription_utils import (
    normalize_segments,
    build_sentences,
    build_groups,
    generate_srt_from_segments,
)

logger = logging.getLogger("transcription-service")


def run_transcription_pipeline(
    video_path: str,
    cookie: str,
    *,
    language_hint: str = "zh",
    extract_images: bool = True,
) -> Dict[str, Any]:
    """
    通用转写流水线（不关心 MQ、不关心任务 ID）：
    输入：
        - video_path : 本地视频文件绝对路径
        - cookie     : 用于访问文件服务
        - language_hint : 语言提示（阿里云失败时兜底）
        - extract_images: 是否为每个 group 抽帧配图
    输出：
        dict = {
            "text": str,
            "language": str,
            "segments": [ {...}, ... ],          # 秒制
            "sentences": [ {...}, ... ],
            "groups": [ {..., "imageUrls": []}, ... ],
            "subtitles": str,                    # SRT 文本
            "videoUrl": str,                     # HLS m3u8 URL
            "duration": str,                     # 00:00:00.000
            "durationSeconds": float,
        }
    """
    audio_path: str | None = None
    audio_url: str | None = None

    # 1. 时长
    duration_seconds = get_media_duration(video_path)
    duration_fmt = format_duration(duration_seconds)
    logger.info(f"[pipeline] 媒体时长: {duration_seconds:.2f}s -> {duration_fmt}")

    # 2. 转 HLS + 上传 m3u8
    logger.info("[pipeline] 开始 HLS 转码 + 上传")
    hls_info = convert_video_to_hls(video_path)
    m3u8_url = upload_hls_files(hls_info, cookie, cleanup=True)
    logger.info(f"[pipeline] HLS 上传完成: {m3u8_url}")

    try:
        # 3. 视频转音频 + 上传
        logger.info("[pipeline] 开始视频转音频 + 上传")
        audio_path = convert_video_to_audio(video_path)
        audio_url = upload_file_to_server(audio_path, cookie, file_type="Taudio")
        logger.info(f"[pipeline] 音频上传完成: {audio_url}")

        # 4. 调阿里云转写
        client = AliyunFileTrans(
            config.get("ALIYUN_ACCESS_KEY_ID"),
            config.get("ALIYUN_ACCESS_KEY_SECRET"),
        )
        aliyun_task_id = client.submit_file_trans_request(
            config.get("ALIYUN_APP_KEY"),
            audio_url,
        )
        if not aliyun_task_id:
            raise Exception("提交录音文件识别请求失败")

        result_json = client.get_file_trans_result(aliyun_task_id)
        if result_json is None:
            raise Exception("识别结果查询失败")

        formatted = client.format_result_for_mq(result_json)
        base_text = formatted.get("text", "") or ""
        base_language = formatted.get("language", "") or language_hint
        raw_segments = formatted.get("segments", []) or []

        # 5. 规范化 segments → 秒制
        segments = normalize_segments(raw_segments)

        # 6. sentence / group / SRT
        sentences = build_sentences(
            segments,
            target_sentence_duration=30.0,
            max_sentence_duration=45.0,
        )
        groups = build_groups(
            sentences,
            target_group_duration=90.0,
            max_group_duration=120.0,
        )
        subtitles = generate_srt_from_segments(segments)

        logger.info(
            f"[pipeline] 分句完成: sentences={len(sentences)}, groups={len(groups)}"
        )

        # 7. group 级配图（可选）
        if extract_images and groups:
            group_image_map = extract_group_best_frames(
                video_path,
                groups,
                sentences,
                cookie,
                max_images_per_group=3,
            )
        else:
            group_image_map = {}

        for g in groups:
            gid = g["groupId"]
            g["imageUrls"] = group_image_map.get(gid, [])

        # 8. 返回统一结构
        return {
            "text": base_text,
            "language": base_language,
            "segments": segments,
            "sentences": sentences,
            "groups": groups,
            "subtitles": subtitles,
            "videoUrl": m3u8_url,
            "duration": duration_fmt,
            "durationSeconds": duration_seconds,
        }

    finally:
        # 清理音频：远程 + 本地
        try:
            if audio_url:
                delete_file_by_url(audio_url, cookie)
                logger.info(f"[pipeline] 已删除远程音频: {audio_url}")
        except Exception as e:
            logger.warning(f"[pipeline] 删除远程音频失败: {e}")

        try:
            import os
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"[pipeline] 已删除本地音频: {audio_path}")
        except Exception as e:
            logger.warning(f"[pipeline] 删除本地音频失败: {e}")
