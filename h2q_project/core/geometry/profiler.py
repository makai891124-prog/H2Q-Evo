import time
import tracemalloc
import functools

class Profiler:
    def __init__(self, name="Profiler"):
        self.name = name
        self.start_time = None
        self.start_memory = None

    def __enter__(self):
        print(f"Entering {self.name}")
        self.start_time = time.time()
        tracemalloc.start()
        self.start_memory = tracemalloc.get_traced_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed_time = end_time - self.start_time
        memory_diff = current_memory - self.start_memory[0] if self.start_memory else current_memory
        peak_memory_diff = peak_memory - self.start_memory[1] if self.start_memory else peak_memory
        print(f"{self.name} took {elapsed_time:.4f} seconds")
        print(f"{self.name} Memory usage: current={current_memory} peak={peak_memory}")
        print(f"{self.name} Memory difference from start: current={memory_diff} peak={peak_memory_diff}")


def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with Profiler(func.__name__):
            return func(*args, **kwargs)
    return wrapper