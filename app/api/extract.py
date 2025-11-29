from fastapi import APIRouter, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import tempfile, os, uuid, logging, json, gc
from app.core.video_utils import (
    extract_frames_sequential, select_best_frame, batch_upload_frames,
    video_cache, parse_time
)

router = APIRouter(tags=["Extract"])

@router.post("/extract")
async def extract_frames(
    video: UploadFile,
    segment: bool = Form(...),
    timestamp: str = Form(...),  # JSON 字符串
    cookie: str = Form(...)
):
    tmp_path = None
    try:
        unique_filename = f"{uuid.uuid4()}.mp4"
        tmp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, unique_filename)

        content = await video.read()
        with open(tmp_path, 'wb') as f:
            f.write(content)

        logging.info(f"临时视频文件: {tmp_path}")

        ts_data = json.loads(timestamp)
        if "timestamp" not in ts_data:
            raise ValueError("缺少 timestamp 字段")

        upload_frames = []
        if segment:
            for seg_idx, seg in enumerate(ts_data["timestamp"]):
                logging.info(f"段落 {seg_idx}, 数量: {len(seg)}")
                frames = extract_frames_sequential(tmp_path, seg)
                best = select_best_frame(frames)
                if best is not None:
                    upload_frames.append(best)
                frames.clear(); gc.collect()
        else:
            upload_frames = extract_frames_sequential(tmp_path, ts_data["timestamp"])

        if not upload_frames:
            raise Exception("未提取到任何图片，无法上传")

        max_batch_size = 5
        result = []
        for i in range(0, len(upload_frames), max_batch_size):
            batch = upload_frames[i:i+max_batch_size]
            logging.info(f"上传批次 {i//max_batch_size+1}, 图片数: {len(batch)}")
            batch_result = batch_upload_frames(batch, cookie)
            result.extend(batch_result)
            import time; time.sleep(0.1)

        upload_frames.clear(); gc.collect()
        result.sort(key=lambda item: parse_time(item["timestamp"]))
        return JSONResponse(content={"urls": result})
    except Exception as e:
        logging.error(f"/extract 异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            video_cache.release_all()
        except Exception as ex:
            logging.error(f"释放视频缓存失败: {ex}")
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logging.info(f"临时文件已删除: {tmp_path}")
            except Exception as e:
                logging.warning(f"删除临时文件失败: {e}")
        gc.collect()
