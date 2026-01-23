import psutil
import time
import logging

class MemoryMonitor:
    def __init__(self, threshold_percent=90, check_interval=5):
        self.threshold_percent = threshold_percent
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)

    def check_memory_usage(self):
        vm = psutil.virtual_memory()
        memory_percent_used = vm.percent

        if memory_percent_used > self.threshold_percent:
            self.logger.warning(f"Memory usage is high: {memory_percent_used:.2f}% > {self.threshold_percent}%.")

    def start_monitoring(self, stop_event):
        self.logger.info("Starting memory monitoring...")
        while not stop_event.is_set():
            self.check_memory_usage()
            time.sleep(self.check_interval)
        self.logger.info("Memory monitoring stopped.")