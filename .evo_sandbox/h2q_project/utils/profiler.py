import time
import psutil
import os
import json

class Profiler:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def duration(self):
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_cpu_usage(self):
        return psutil.cpu_percent()

    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # in MB

    def get_gpu_usage(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            pynvml.nvmlShutdown()
            return gpu_util
        except Exception as e:
            print(f"GPU usage monitoring not available: {e}")
            return None

    def get_profile_data(self):
      profile_data = {
          "name": self.name,
          "duration": self.duration(),
          "cpu_usage": self.get_cpu_usage(),
          "memory_usage": self.get_memory_usage()
      }
      gpu_usage = self.get_gpu_usage()
      if gpu_usage is not None:
        profile_data["gpu_usage"] = gpu_usage

      return profile_data

    def print_profile_data(self):
        profile_data = self.get_profile_data()
        print(json.dumps(profile_data, indent=4))


if __name__ == '__main__':
    # Example usage:
    with Profiler("Test") as profiler:
        time.sleep(1)

    profiler.print_profile_data()
