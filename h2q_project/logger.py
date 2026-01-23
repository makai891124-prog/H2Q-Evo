import json
import os
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, log_file='training_log.json'):
        self.log_file = log_file
        self.data = []

    def log(self, metrics):
        self.data.append(metrics)
        self._save_log()

    def _save_log(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.data, f, indent=4)

    def load_log(self):
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                try:
                    self.data = json.load(f)
                except json.JSONDecodeError:
                    self.data = []  # Handle empty or corrupted log files
        else:
            self.data = []

    def visualize(self, metric_name, output_file='metric_plot.png'):
        if not self.data or not self.data[0].get(metric_name):
            print(f"Metric '{metric_name}' not found in log data.")
            return

        epochs = range(1, len(self.data) + 1)
        metric_values = [entry[metric_name] for entry in self.data]

        plt.plot(epochs, metric_values)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'Training {metric_name}')
        plt.grid(True)
        plt.savefig(output_file)
        plt.show()

if __name__ == '__main__':
    # Example Usage
    logger = Logger()

    # Simulate training loop
    for epoch in range(1, 6):
        metrics = {
            'epoch': epoch,
            'loss': 1.0 / epoch,
            'accuracy': 0.2 * epoch
        }
        logger.log(metrics)
        print(f"Epoch {epoch}: Loss = {metrics['loss']:.4f}, Accuracy = {metrics['accuracy']:.4f}")

    # Visualize the loss
    logger.visualize('loss')
    logger.visualize('accuracy')

    # Load existing logs and visualise them
    logger2 = Logger()
    logger2.load_log()
    logger2.visualize('loss', 'loaded_loss.png')