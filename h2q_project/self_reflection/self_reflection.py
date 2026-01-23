class SelfReflection:
    def __init__(self, kernel):
        self.kernel = kernel

    def reflect_on_creation(self, object_type, object_data):
        message = f"Reflecting on creation of {object_type} with data: {object_data}"
        return self.kernel.reflect(message)

    def get_reflection_log(self):
        return self.kernel.get_reflection_log()
