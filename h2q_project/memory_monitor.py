import psutil
import logging
import time

logger = logging.getLogger(__name__)

class MemoryMonitor:
    def __init__(self, interval=60):
        self.interval = interval

    def start(self):
        while True:
            self.log_memory_usage()
            time.sleep(self.interval)

    def log_memory_usage(self):
        memory = psutil.virtual_memory()
        logger.info(f"Memory Usage: Total={memory.total}, Available={memory.available}, Percent={memory.percent}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    monitor = MemoryMonitor()
    monitor.start()