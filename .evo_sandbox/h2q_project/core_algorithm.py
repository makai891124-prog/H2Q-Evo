from h2q_project.memory_profiler import MemoryProfiler

class CoreAlgorithm:
    def __init__(self):
        self.profiler = MemoryProfiler()

    @MemoryProfiler().profile
    def run_algorithm(self, data):
        # Simulate a memory-intensive operation
        processed_data = [x * 2 for x in data]
        return processed_data

if __name__ == '__main__':
    algorithm = CoreAlgorithm()
    data = list(range(10000))
    result = algorithm.run_algorithm(data)
    print(f"Algorithm completed. First 10 results: {result[:10]}")
