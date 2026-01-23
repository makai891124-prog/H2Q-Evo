import memory_profiler
import time


class MemoryOptimizer:
    def __init__(self, target_function, *args, **kwargs):
        self.target_function = target_function
        self.args = args
        self.kwargs = kwargs
        self.memory_usage = None

    def run_and_profile(self):
        # Warm up the function
        self.target_function(*self.args, **self.kwargs)
        time.sleep(0.1) #Give some time for garbage collector

        # Profile the memory usage
        self.memory_usage = memory_profiler.memory_usage(
            (self.target_function, self.args, self.kwargs),
            max_usage=True, retval=True
        )

        return self.memory_usage

    def get_memory_usage(self):
        return self.memory_usage


# Example usage:
# def my_function(data):
#     my_list = [i for i in range(data)]
#     return my_list
#
# optimizer = MemoryOptimizer(my_function, 1000000)
# memory_usage, return_value = optimizer.run_and_profile()
# print(f"Memory Usage: {memory_usage} MB")
# print(f"Return Value: {return_value[:10]}...") #print first 10 elements