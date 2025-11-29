from app.config import config
import redis

# Redis 客户端单例
redis_client = redis.Redis(
    host=config.get("REDIS_HOST"),
    port=config.get("REDIS_PORT"),
    password=config.get("REDIS_PASSWORD") or None,
    db=config.get("REDIS_DB"),
    decode_responses=True
)
