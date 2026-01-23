from h2q_project.core.data_structures import BloomFilter, LimitedSizeDict
from h2q_project.utils.hashing import fnv1a_hash
from h2q_project.utils.memory_profiler import MemoryProfiler


class H2QCore:
    def __init__(self, bloom_filter_size=1000, cache_size=100, enable_memory_profiling=False):
        self.bloom_filter = BloomFilter(bloom_filter_size, [fnv1a_hash])
        self.cache = LimitedSizeDict(size_limit=cache_size)
        self.memory_profiler = MemoryProfiler(enabled=enable_memory_profiling)

    def process_query(self, query):
        self.memory_profiler.start()
        if query in self.cache:
            result = self.cache[query]
            self.memory_profiler.stop(message=f"Cache hit for query: {query}")
            return result

        if self.bloom_filter.check(query):
            # Simulate a database lookup (expensive operation)
            result = f"Result for {query} from database"
        else:
            result = f"No result found for {query}"

        self.bloom_filter.add(query)
        self.cache[query] = result
        self.memory_profiler.stop(message=f"Cache miss for query: {query}")
        return result