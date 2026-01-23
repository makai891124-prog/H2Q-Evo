import logging
import time

class SystemMonitor:
    def __init__(self, log_level=logging.INFO):
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()

    def log_status(self, message):
        self.logger.info(message)

    def log_error(self, message):
        self.logger.error(message)

    def log_performance(self, metric, value):
        self.logger.info(f'Performance Metric: {metric} = {value}')

    def get_uptime(self):
        uptime = time.time() - self.start_time
        return uptime

    def log_uptime(self):
        uptime = self.get_uptime()
        self.log_status(f'System Uptime: {uptime:.2f} seconds')

# Example Usage (can be moved to a separate script or integrated into existing modules)
if __name__ == '__main__':
    monitor = SystemMonitor()
    monitor.log_status('System started successfully.')
    time.sleep(2)  # Simulate some work
    monitor.log_performance('CPU Usage', 75.5)
    monitor.log_uptime()
    monitor.log_error('An example error occurred.')
