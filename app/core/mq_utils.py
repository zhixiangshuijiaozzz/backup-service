import pika, json, time, logging
from app.config import config

logger = logging.getLogger("mq")

def create_rabbitmq_connection():
    max_retries = 5
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=config.get("RABBITMQ_HOST"),
                    port=config.get("RABBITMQ_PORT"),
                    virtual_host=config.get("RABBITMQ_VHOST"),
                    credentials=pika.PlainCredentials(config.get("RABBITMQ_USER"), config.get("RABBITMQ_PASS")),
                    heartbeat=15,
                    blocked_connection_timeout=300,
                    socket_timeout=10,
                    stack_timeout=15
                )
            )
            logger.info("RabbitMQ 连接成功")
            return connection
        except Exception as e:
            logger.error(f"连接失败 {attempt+1}/5: {e}")
            if attempt<max_retries-1:
                time.sleep(retry_delay)
            else:
                raise

def setup_rabbitmq_channel(connection, exchange, queue, routing_key, is_result_queue=False):
    ch = connection.channel()
    ch.exchange_declare(exchange=exchange, exchange_type='direct', durable=True)
    queue_args = {'x-dead-letter-exchange':'task.dlx.exchange','x-dead-letter-routing-key':'task.dlx','x-message-ttl':3600000}
    ch.queue_declare(queue=queue, durable=True, arguments=queue_args)
    ch.queue_bind(exchange=exchange, queue=queue, routing_key=routing_key)
    if not is_result_queue:
        ch.basic_qos(prefetch_count=1)
    return ch

def send_result_to_exchange(channel, exchange, routing_key, message: dict, headers: dict | None = None):
    channel.basic_publish(
        exchange=exchange,
        routing_key=routing_key,
        body=json.dumps(message),
        properties=pika.BasicProperties(
            delivery_mode=2,
            headers=headers or {}
        )
    )
    logger.info(f"结果已发送: taskId={message.get('taskId')}")
