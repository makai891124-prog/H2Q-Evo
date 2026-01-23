import psutil
import os


def get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss  # Resident Set Size


class MemoryProfiler:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.start_memory = 0

    def start(self):
        if self.enabled:
            self.start_memory = get_process_memory()

    def stop(self, message="Memory usage:"):
        if self.enabled:
            end_memory = get_process_memory()
            memory_used = end_memory - self.start_memory
            print(f"{message} {memory_used / 1024 / 1024:.2f} MB")
