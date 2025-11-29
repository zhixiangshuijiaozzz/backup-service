from app.deps import redis_client
from app.config import config
import logging

logger = logging.getLogger("redis")

def lock_task_in_redis(task_id: str) -> bool:
    """检查并将任务 key 从 task:{id} 改为 task:convert@{id}，实现幂等与占位"""
    prefix = config.get("REDIS_TASK_KEY_PREFIX","task:")
    ttl    = config.get("REDIS_TASK_TTL",3600)
    redis_key = f"{prefix}{task_id}"
    convert_key = f"{prefix}convert@{task_id}"

    try:
        exists = redis_client.exists(redis_key)
        logger.info(f"检查任务 {task_id} 是否存在: {'存在' if exists else '不存在'}")
        if exists:
            task_data = redis_client.get(redis_key)
            redis_client.set(convert_key, task_data or "1", ex=ttl)
            redis_client.delete(redis_key)
            logger.info(f"已标记处理中: {convert_key} 并移除 {redis_key}")
        return bool(exists)
    except Exception as e:
        logger.error(f"Redis 错误: {e}")
        return False
