# app/core/video_utils.py
import os
import io
import gc
import json
import uuid
import logging
import subprocess
import time
from typing import List, Tuple, Union, Dict

import cv2
import numpy as np
import requests

from app.config import config

logger = logging.getLogger("video-utils")

# ============ 视频缓存 ============
class VideoCache:
    def __init__(self, max_size=5):
        self.cache: Dict[str, cv2.VideoCapture] = {}
        self.max_size = max_size
        self.usage_order: List[str] = []
        import threading
        self.lock = threading.Lock()

    def get_video(self, path: str) -> cv2.VideoCapture:
        with self.lock:
            if path in self.cache:
                if not self.cache[path].isOpened():
                    self.cache[path] = cv2.VideoCapture(path)
                if path in self.usage_order:
                    self.usage_order.remove(path)
                self.usage_order.append(path)
                return self.cache[path]

            if len(self.cache) >= self.max_size and self.usage_order:
                oldest = self.usage_order.pop(0)
                if oldest in self.cache:
                    try:
                        self.cache[oldest].release()
                    except Exception as e:
                        logger.error(f"释放视频出错: {e}")
                    del self.cache[oldest]

            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                self.cache[path] = cap
                self.usage_order.append(path)
            return cap

    def release_all(self):
        with self.lock:
            for path, cap in list(self.cache.items()):
                try:
                    cap.release()
                except Exception:
                    pass
                self.cache.pop(path, None)
            self.usage_order.clear()


video_cache = VideoCache()

# ============ 时间解析工具 ============
def parse_time(time_str: str) -> float:
    s = time_str.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
        elif len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
        else:
            raise ValueError("不支持的时间格式")
    return float(s)


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ============ 帧提取 ============
def extract_frame(video_path: str, timestamp: float) -> Union[np.ndarray, None]:
    try:
        cap = video_cache.get_video(video_path)
        if not cap or not cap.isOpened():
            logger.error("无法打开视频文件")
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"{timestamp}s 未提取到帧")
            return None
        if frame is not None and frame.shape[0] > 720:
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        return frame
    except Exception as e:
        logger.error(f"提取帧出错: {e}")
        return None


def calculate_frame_complexity(image: np.ndarray) -> float:
    """
    综合边缘密度 / 对比度 / 非黑像素比例 / 锐度 / 熵
    用来衡量“内容丰富度 + 文本丰富度”的近似指标
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.count_nonzero(edges) / edges.size
        std_dev = np.std(gray)
        non_black_ratio = np.count_nonzero(gray > 10) / gray.size
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(lap)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-7)
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        score = (
            edge_density * 0.3
            + (std_dev / 128) * 0.2
            + non_black_ratio * 0.2
            + min(1.0, sharpness / 1000) * 0.15
            + (entropy / 8) * 0.15
        )
        return max(0.0, min(1.0, score))
    except Exception as e:
        logger.error(f"复杂度计算出错: {e}")
        return 0.0


def is_normal_frame(image: np.ndarray) -> bool:
    """
    过滤黑屏 / 白屏 / 过度平滑闪烁帧。
    - 均值太低/太高视为黑/白屏；
    - 标准差太低视为几乎全一色。
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))

        # 过暗 or 过亮
        if mean_val < 10 or mean_val > 245:
            return False

        # 对比度太低（接近纯色）
        if std_val < 5:
            return False

        return True
    except Exception as e:
        logger.error(f"帧有效性判断异常: {e}")
        return False


def extract_frames_sequential(video_path: str, timestamps: List[str]) -> List[Tuple[str, np.ndarray]]:
    results = []
    for ts in timestamps:
        try:
            f = extract_frame(video_path, parse_time(ts))
            if f is not None:
                results.append((ts, f))
        except Exception as e:
            logger.error(f"处理时间戳 {ts} 出错: {e}")
    return results


def select_best_frame(frames: List[Tuple[str, np.ndarray]]) -> Union[Tuple[str, np.ndarray], None]:
    if not frames:
        return None
    best_score, best_pair = -1.0, None
    for ts, frame in frames:
        try:
            sc = calculate_frame_complexity(frame)
            logger.info(f"{ts} 复杂度: {sc:.4f}")
            if sc > best_score:
                best_score, best_pair = sc, (ts, frame)
        except Exception as e:
            logger.error(f"评分异常: {e}")
    return best_pair


# ============ 感知 Hash & 去重 ============
def compute_ahash(image: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """
    平均哈希（aHash），用于判定两帧是否“长得差不多”。
    返回 0/1 numpy 数组，长度 hash_size * hash_size。
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        resized = cv2.resize(gray, (hash_size, hash_size))
        avg = resized.mean()
        diff = resized > avg
        return diff.astype(np.uint8).flatten()
    except Exception as e:
        logger.error(f"aHash 计算失败: {e}")
        # 返回全 0，保证出错时 hash 差异较小
        return np.zeros((hash_size * hash_size,), dtype=np.uint8)


def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    if h1.shape != h2.shape:
        # 长度不同时，直接认为差异很大
        return 64
    return int(np.count_nonzero(h1 ^ h2))


def pick_diverse_frames(
    frames_with_ts: List[Tuple[str, np.ndarray]],
    max_count: int = 3,
    dup_threshold: int = 5,
    hash_size: int = 8,
) -> List[Tuple[str, np.ndarray]]:
    """
    在若干帧中选出 1~max_count 张：
    - 复杂度评分高；
    - 互相的感知 Hash 距离大于 dup_threshold，避免选到几乎相同的帧。
    """
    if not frames_with_ts:
        return []

    candidates = []
    for ts, frame in frames_with_ts:
        if frame is None:
            continue
        if not is_normal_frame(frame):
            logger.info(f"丢弃异常帧（黑屏/白屏/低对比度）: {ts}")
            continue
        score = calculate_frame_complexity(frame)
        ah = compute_ahash(frame, hash_size=hash_size)
        candidates.append({"ts": ts, "frame": frame, "score": score, "hash": ah})

    if not candidates:
        return []

    # 按分数降序
    candidates.sort(key=lambda x: x["score"], reverse=True)

    selected: List[Tuple[str, np.ndarray]] = []
    selected_hashes: List[np.ndarray] = []

    for item in candidates:
        if len(selected) >= max_count:
            break
        ts = item["ts"]
        frame = item["frame"]
        h = item["hash"]

        is_dup = False
        for sh in selected_hashes:
            if hamming_distance(h, sh) <= dup_threshold:
                is_dup = True
                break
        if is_dup:
            logger.info(f"丢弃相似帧: {ts}")
            continue

        selected.append((ts, frame))
        selected_hashes.append(h)

    return selected


# ============ 文件上传/删除 ============
def upload_file_to_server(file_path, cookie, file_type: str = ""):
    logger.info(f"上传文件: {file_path}")
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        headers = {"Cookie": cookie}
        params = {"type": file_type}
        r = requests.post(config.get("FILE_UPLOAD_URL"), files=files, headers=headers, params=params)
        if r.status_code == 200:
            url = r.text.strip('"')
            logger.info(f"上传成功: {url}")
            return url
        raise Exception(f"上传失败: {r.status_code} {r.text}")


def delete_file_by_url(url, cookie):
    logger.info(f"删除文件: {url}")
    headers = {"Cookie": cookie}
    r = requests.delete(f"{config.get('FILE_DELETE_URL')}?url={url}", headers=headers)
    return r.status_code == 200


def batch_upload_frames(frames_with_ts: List[Tuple[str, np.ndarray]], cookie: str) -> List[dict]:
    files = []
    file_bytes_list = []
    for idx, (ts, frame) in enumerate(frames_with_ts):
        is_ok, buffer = cv2.imencode(".jpg", frame)
        if not is_ok:
            logger.warning(f"编码失败: {ts}")
            continue
        bio = io.BytesIO(buffer.tobytes())
        bio.seek(0)
        filename = f"frame_{idx}_{int(cv2.getTickCount())}.jpg"
        files.append(("files", (filename, bio, "image/jpeg")))
        file_bytes_list.append(bio)
    if not files:
        raise Exception("没有可上传的图片")

    r = requests.post(config.get("FILE_MULTI_UPLOAD_URL"), files=files, headers={"Cookie": cookie})
    try:
        if r.headers.get("content-type", "").startswith("application/json"):
            data = r.json()
        else:
            data = json.loads(r.text)
    except Exception:
        data = r.text.split(",")

    if r.status_code != 200:
        raise Exception(f"批量上传失败: {r.status_code} {r.text}")

    urls = data if isinstance(data, list) else data.get("urls", [])
    if len(urls) != len(frames_with_ts):
        raise Exception(f"返回URL数量({len(urls)})与上传数量({len(frames_with_ts)})不一致")

    result = [{"timestamp": ts, "url": url} for (ts, _), url in zip(frames_with_ts, urls)]
    file_bytes_list.clear()
    gc.collect()
    return result


# ============ 媒体处理（ffmpeg / HLS） ============
def get_media_duration(file_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
    return float(out)


def has_video_stream(file_path: str) -> bool:
    """
    判断媒体文件是否包含视频流。

    如果命令执行异常则默认返回 False，避免影响后续流程。
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            file_path,
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return bool(out)
    except Exception as e:
        logger.warning(f"检测视频流失败: {e}")
        return False


def convert_video_to_audio(video_path: str) -> str:
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return audio_path


def convert_video_to_hls(video_path: str) -> dict:
    output_dir = os.path.join(os.path.dirname(video_path), f"hls_{uuid.uuid4().hex[:8]}")
    os.makedirs(output_dir, exist_ok=True)
    m3u8_path = os.path.join(output_dir, "index.m3u8")
    ts_pattern = os.path.join(output_dir, "segment_%03d.ts")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-f",
        "hls",
        "-hls_time",
        str(config.get("HLS_SEGMENT_TIME", 6)),
        "-hls_playlist_type",
        "vod",
        "-hls_segment_filename",
        ts_pattern,
        "-hls_list_size",
        "0",
        m3u8_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    if not os.path.exists(m3u8_path):
        logger.error(f"m3u8 未生成: {m3u8_path}")
        raise Exception("m3u8 未创建")
    ts_files = sorted(
        [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".ts")]
    )
    return {"m3u8_path": m3u8_path, "ts_files": ts_files, "output_dir": output_dir}


def upload_hls_files(hls_data, cookie, cleanup: bool = True) -> str:
    """
    上传所有 ts 文件 + 修改后的 m3u8，
    返回 m3u8 的 URL，并根据 cleanup 选择是否删除本地 HLS 目录。
    """
    import shutil

    m3u8_path = hls_data["m3u8_path"]
    ts_files = hls_data["ts_files"]
    for ts in ts_files:
        if not os.path.exists(ts):
            raise FileNotFoundError(f"ts 不存在: {ts}")

    files_data = []
    for ts_file in ts_files:
        with open(ts_file, "rb") as f:
            content = f.read()
        files_data.append(("files", (os.path.basename(ts_file), content, "video/mp2t")))

    headers = {"Cookie": cookie}
    multi_resp = requests.post(
        config.get("FILE_MULTI_UPLOAD_URL"), files=files_data, headers=headers, params={"type": "Tvideo"}
    )
    if multi_resp.status_code != 200:
        raise Exception(f"ts 批量上传失败: {multi_resp.status_code} {multi_resp.text}")
    ts_urls = multi_resp.json()

    with open(m3u8_path, "r") as f:
        m3u8_content = f.read()
    for i, ts_file in enumerate(ts_files):
        m3u8_content = m3u8_content.replace(os.path.basename(ts_file), ts_urls[i])

    modified_m3u8_path = os.path.join(os.path.dirname(m3u8_path), "modified_index.m3u8")
    with open(modified_m3u8_path, "w") as f:
        f.write(m3u8_content)

    with open(modified_m3u8_path, "rb") as f:
        files = {"file": ("index.m3u8", f, "application/vnd.apple.mpegurl")}
        resp = requests.post(
            config.get("FILE_UPLOAD_URL"), files=files, headers=headers, params={"type": "Tvideo"}
        )
    if resp.status_code != 200:
        raise Exception(f"m3u8 上传失败: {resp.status_code} {resp.text}")
    m3u8_url = resp.text.strip('"')

    if cleanup:
        try:
            shutil.rmtree(os.path.dirname(m3u8_path), ignore_errors=True)
        except Exception as e:
            logger.warning(f"HLS 临时目录删除失败: {e}")

    return m3u8_url


# ============ group 级别的图片提取辅助 ============
def extract_group_candidate_frames(
    video_path: str,
    group: Dict[str, any],
    sentences: Dict[str, Dict[str, any]],
) -> List[Tuple[str, np.ndarray]]:
    """
    按 group 内 sentences 的时间范围，在 1/3 和 2/3 处取帧。
    group: {"groupId":..., "sentenceIds":[...], ...}
    sentences: {sentenceId: {...}}
    返回: [(timestamp_str, frame), ...]
    """
    candidates: List[Tuple[str, np.ndarray]] = []
    for sid in group.get("sentenceIds", []):
        sent = sentences.get(sid)
        if not sent:
            continue
        start = float(sent["start"])
        end = float(sent["end"])
        if end <= start:
            continue
        duration = end - start
        # 1/3 和 2/3 位置
        t1 = start + duration / 3.0
        t2 = start + duration * 2.0 / 3.0
        for t in [t1, t2]:
            frame = extract_frame(video_path, t)
            if frame is None:
                continue
            ts_str = format_duration(t)
            candidates.append((ts_str, frame))
    return candidates


def extract_group_best_frames(
    video_path: str,
    groups: List[Dict[str, any]],
    sentence_list: List[Dict[str, any]],
    cookie: str,
    max_images_per_group: int = 3,
) -> Dict[str, List[str]]:
    """
    核心逻辑：
    - 对每个 group，按 sentence 的 1/3 & 2/3 时间点取帧；
    - 过滤异常帧 / 黑白屏；
    - 按复杂度评分 + 感知 hash 选 1~3 张；
    - 批量上传，返回 groupId -> [imageUrl,...] 映射。
    """
    # 建一个 sentenceId -> sentence 的索引
    sentence_map = {s["sentenceId"]: s for s in sentence_list}
    group_to_urls: Dict[str, List[str]] = {}

    for group in groups:
        gid = group["groupId"]
        logger.info(f"为 group={gid} 提取候选帧")
        candidates = extract_group_candidate_frames(video_path, group, sentence_map)
        if not candidates:
            group_to_urls[gid] = []
            continue

        # 在本 group 内部做“去黑白屏 + 去重 + 选前 1~3”
        selected_frames = pick_diverse_frames(
            candidates,
            max_count=max_images_per_group,
            dup_threshold=5,
            hash_size=8,
        )
        if not selected_frames:
            group_to_urls[gid] = []
            continue

        # 上传当前 group 的所有选中图片
        logger.info(f"group={gid} 最终入库图片张数: {len(selected_frames)}")
        upload_result = batch_upload_frames(selected_frames, cookie)
        # upload_result: [{"timestamp": ts, "url": url},...]
        urls = [item["url"] for item in upload_result]
        group_to_urls[gid] = urls

        # 主动释放临时帧
        candidates.clear()
        selected_frames.clear()
        gc.collect()

    return group_to_urls
