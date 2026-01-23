class BloomFilter:
    def __init__(self, size, hash_functions):
        self.size = size
        self.bit_array = [False] * size
        self.hash_functions = hash_functions

    def add(self, item):
        for hash_function in self.hash_functions:
            index = hash_function(item) % self.size
            self.bit_array[index] = True

    def check(self, item):
        for hash_function in self.hash_functions:
            index = hash_function(item) % self.size
            if not self.bit_array[index]:
                return False
        return True


class SimpleCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.access_order = []

    def get(self, key):
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        self.cache[key] = value
        self.access_order.append(key)


class LimitedSizeDict(object):
    def __init__(self, size_limit=1000):
        self.size_limit = size_limit
        self.data = {}
        self.keys = []

    def __setitem__(self, key, value):
        if key not in self.data:
            if len(self.data) >= self.size_limit:
                oldest = self.keys.pop(0)
                del self.data[oldest]
            self.keys.append(key)
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def clear(self):
        self.data = {}
        self.keys = []
