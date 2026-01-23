import time

class GeometryKernel:
    def __init__(self):
        self.operations_count = 0

    def perform_operation(self, operation_type, data):
        start_time = time.time()
        result = self._execute_operation(operation_type, data)
        end_time = time.time()

        self.operations_count += 1
        duration = end_time - start_time
        print(f"Operation '{operation_type}' took {duration:.4f} seconds.")

        return result

    def _execute_operation(self, operation_type, data):
        # Simulate different geometry operations with varying complexity
        if operation_type == "intersection":
            # Simulate a complex intersection calculation
            time.sleep(0.01)  # Simulate work
            return "Intersection Result"
        elif operation_type == "distance":
            # Simulate a distance calculation
            time.sleep(0.005) # Simulate work
            return "Distance Result"
        elif operation_type == "area":
            #Simulate area calculation
            time.sleep(0.001)
            return "Area result"
        else:
            return None

    def get_operations_count(self):
        return self.operations_count