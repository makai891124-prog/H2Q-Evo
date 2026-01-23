import logging

class Workflow:
    def __init__(self, steps):
        self.steps = steps
        self.logger = logging.getLogger(__name__)

    def run(self, data):
        self.logger.info("Workflow started")
        try:
            for step in self.steps:
                data = step.execute(data)
            self.logger.info("Workflow completed successfully")
            return data
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}", exc_info=True)
            raise

class Step:
    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger(__name__)

    def execute(self, data):
        self.logger.info(f"Executing step: {self.name}")
        raise NotImplementedError("Execute method must be implemented in subclass")

# Example usage (can be moved to a separate test file)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    class ExampleStep(Step):
        def __init__(self, name, value_to_add: int):
            super().__init__(name)
            self.value_to_add = value_to_add

        def execute(self, data: int) -> int:
            self.logger.info(f"Adding {self.value_to_add} to {data}")
            result = data + self.value_to_add
            if result < 0:
                self.logger.warning("Result is negative!")
            return result

    # Example steps with type annotations and numerical calculation
    step1 = ExampleStep("Add 5", 5)
    step2 = ExampleStep("Add -10", -10)

    # Example workflow
    workflow = Workflow([step1, step2])

    # Example data
    initial_data = 10

    # Run the workflow
    try:
        final_data = workflow.run(initial_data)
        print(f"Final data: {final_data}")
    except Exception as e:
        print(f"Workflow failed: {e}")
