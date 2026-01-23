import gc

class MemoryTracker:
    def __init__(self):
        self.initial_memory = self.get_memory_usage()

    def get_memory_usage(self):
        gc.collect()
        return len(gc.get_objects())

    def check_leaks(self):
        current_memory = self.get_memory_usage()
        leak_detected = current_memory > self.initial_memory
        if leak_detected:
            delta = current_memory - self.initial_memory
            print(f"Memory leak detected: {delta} objects")
        return leak_detected