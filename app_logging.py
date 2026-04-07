import logging
import logging.handlers
import queue
import threading

log_queue = queue.Queue()

queue_handler = logging.handlers.QueueHandler(log_queue)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
logger.addHandler(queue_handler)

def log_worker():
    while True:
        record = log_queue.get()
        if record is None:
            break
        handler.handle(record)

thread = threading.Thread(target=log_worker, daemon=True)
thread.start()