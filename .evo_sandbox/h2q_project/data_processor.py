import time
from h2q_project.resource_manager import ResourceManager

class DataProcessor:
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager

    def process_data(self, data_id):
        resource = self.resource_manager.acquire_resource(data_id)
        print(f"Processing data with resource {resource.id}")
        time.sleep(0.1)
        self.resource_manager.release_resource(data_id)