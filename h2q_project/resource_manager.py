class ResourceManager:
    def __init__(self):
        self.resources = {}

    def acquire_resource(self, resource_id):
        if resource_id not in self.resources:
            self.resources[resource_id] = Resource(resource_id)
        return self.resources[resource_id]

    def release_resource(self, resource_id):
        if resource_id in self.resources:
            del self.resources[resource_id]


class Resource:
    def __init__(self, resource_id):
        self.id = resource_id

    def __del__(self):
        print(f"Resource {self.id} destroyed")