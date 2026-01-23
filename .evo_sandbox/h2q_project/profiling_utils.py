import time
import tracemalloc

class MemoryProfiler:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        tracemalloc.start()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        current, peak = tracemalloc.get_traced_memory()
        end_time = time.time() - self.start_time
        print(f"Memory profiling for {self.name}:")
        print(f"  Current memory usage: {current / 10**6:.2f} MB")
        print(f"  Peak memory usage: {peak / 10**6:.2f} MB")
        print(f"  Execution time: {end_time:.4f} seconds")
        tracemalloc.stop()