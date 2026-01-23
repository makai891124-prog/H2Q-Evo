import numpy as np
import gc

class ModelTrainer:
    def __init__(self, features, labels):
        self.features = np.array(features) # Use numpy arrays
        self.labels = np.array(labels)
        self.model = None

    def train_model(self, epochs=10):
        # Simulate model training (replace with actual training logic)
        self.model = np.random.rand(self.features.shape[1], self.labels.shape[1]) if len(self.labels.shape) > 1 else np.random.rand(self.features.shape[1]) #Creating random weights
        for epoch in range(epochs):
            # Example: Update model weights (replace with actual training algorithm)
            predictions = self.predict(self.features)
            error = predictions - self.labels
            # Simple weight update (example)
            self.model -= 0.01 * np.dot(self.features.T, error) if len(self.labels.shape) > 1 else 0.01 * np.mean(error)
            print(f'Epoch {epoch + 1}/{epochs}')

        return self.model

    def predict(self, features):
      return np.dot(features, self.model) if len(self.labels.shape) > 1 else features * self.model # return prediction with numpy

    def clear_data(self):
        # Explicitly release memory
        self.features = None
        self.labels = None
        self.model = None
        gc.collect()

# Example Usage (Illustrative)
if __name__ == '__main__':
    # Simulate a dataset
    num_samples = 1000
    num_features = 10
    features = np.random.rand(num_samples, num_features).tolist() # convert to list to simulate the data
    labels = np.random.rand(num_samples, 1).tolist() if num_features > 1 else np.random.rand(num_samples).tolist()

    trainer = ModelTrainer(features, labels)
    model = trainer.train_model()
    print(f"Trained model: {model}")

    trainer.clear_data()
    del features, labels, trainer, model
    gc.collect()
