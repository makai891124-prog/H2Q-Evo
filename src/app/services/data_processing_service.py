import time
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataProcessingService:
    def __init__(self, data_source):
        self.data_source = data_source
        self.cache = {}

    def fetch_and_process_data(self, key: str) -> Dict:
        """Fetches data from the data source and processes it."""
        # Check if data is in cache
        if key in self.cache:
            logging.info(f"Data for key '{key}' found in cache.")
            return self.cache[key]

        logging.info(f"Fetching data for key '{key}' from data source.")
        start_time = time.time()
        data = self.data_source.fetch_data(key)
        fetch_time = time.time() - start_time
        logging.info(f"Data fetch for key '{key}' took {fetch_time:.4f} seconds.")

        if data is None:
            logging.warning(f"No data found for key '{key}'.")
            return {}

        start_time = time.time()
        processed_data = self._process_data(data)
        processing_time = time.time() - start_time
        logging.info(f"Data processing for key '{key}' took {processing_time:.4f} seconds.")

        # Store the processed data in the cache
        self.cache[key] = processed_data

        return processed_data

    def _process_data(self, data: List[Dict]) -> Dict:
        """Processes the fetched data.  Identified as potential bottleneck.
           Optimized by avoiding unnecessary object creation and direct accumulation.
        """
        # Original implementation (inefficient):
        # result = {}
        # for item in data:
        #     for k, v in item.items():
        #         if k in result:
        #             result[k] += v
        #         else:
        #             result[k] = v
        # return result

        # Optimized implementation (more efficient):
        result = {}
        for item in data:
            for k, v in item.items():
                result[k] = result.get(k, 0) + v
        return result



class DataSource:
    """A dummy data source for demonstration purposes."""
    def __init__(self, data: Dict[str, List[Dict]]):
        self.data = data

    def fetch_data(self, key: str) -> List[Dict]:
        """Simulates fetching data from a data source."""
        time.sleep(0.1)  # Simulate network latency
        return self.data.get(key)


if __name__ == '__main__':
    # Example Usage
    data_source = DataSource({
        "key1": [{
            "a": 1, "b": 2, "c": 3
        }, {
            "a": 4, "b": 5, "c": 6
        }],
        "key2": [{
            "x": 7, "y": 8, "z": 9
        }, {
            "x": 10, "y": 11, "z": 12
        }]
    })

    data_processing_service = DataProcessingService(data_source)

    # Process data for key1
    result1 = data_processing_service.fetch_and_process_data("key1")
    print(f"Processed data for key1: {result1}")

    # Process data for key2
    result2 = data_processing_service.fetch_and_process_data("key2")
    print(f"Processed data for key2: {result2}")

    # Access data from cache (faster)
    result1_cached = data_processing_service.fetch_and_process_data("key1")
    print(f"Cached data for key1: {result1_cached}")