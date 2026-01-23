import tracemalloc
import time
import functools

class MemoryProfiler:
    def __init__(self, top_n=10):
        self.top_n = top_n
        self.snapshot = None

    def start(self):
        tracemalloc.start()
        self.snapshot = tracemalloc.take_snapshot()

    def stop(self):
        if self.snapshot is None:
            print("Profiler not started.")
            return

        snapshot2 = tracemalloc.take_snapshot()
        top_stats = snapshot2.compare_to(self.snapshot, 'lineno')

        print("[ Top {} differences ]".format(self.top_n))
        for stat in top_stats[:self.top_n]:
            print(stat)

        self.snapshot = None
        tracemalloc.stop()

    def profile(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.start()
            result = func(*args, **kwargs)
            self.stop()
            return result
        return wrapper

if __name__ == '__main__':
    # Example usage
    @MemoryProfiler().profile
    def allocate_memory(size):
        data = [i for i in range(size)]
        return data

    allocate_memory(100000)
