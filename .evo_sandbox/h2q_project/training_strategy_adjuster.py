class TrainingStrategyAdjuster:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def decrease_learning_rate(self, factor=0.1):
        self.learning_rate *= factor
        print(f"Learning rate decreased to: {self.learning_rate}")

    def increase_learning_rate(self, factor=2.0):
        self.learning_rate *= factor
        print(f"Learning rate increased to: {self.learning_rate}")

    def get_learning_rate(self):
        return self.learning_rate