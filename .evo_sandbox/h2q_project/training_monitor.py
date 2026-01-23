import tensorflow as tf
import os

class TrainingMonitor:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def record_scalar(self, tag, value, step):
        with self.summary_writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.summary_writer.flush()

    def record_histogram(self, tag, values, step):
        with self.summary_writer.as_default():
            tf.summary.histogram(tag, values, step=step)
            self.summary_writer.flush()

    def record_image(self, tag, image, step):
        with self.summary_writer.as_default():
            tf.summary.image(tag, image, step=step, max_outputs=1)
            self.summary_writer.flush()


if __name__ == '__main__':
    # Example Usage
    log_dir = 'logs/example'
    monitor = TrainingMonitor(log_dir)

    for step in range(10):
        monitor.record_scalar('loss', 0.1 * (10 - step), step)
        monitor.record_scalar('accuracy', 0.1 * step, step)

        # Example histogram (replace with actual data)
        import numpy as np
        histogram_data = np.random.rand(100)
        monitor.record_histogram('weights', histogram_data, step)

        # Example image (replace with actual image data)
        image_data = np.random.rand(1, 28, 28, 1)  # Example: grayscale image
        monitor.record_image('input_image', image_data, step)

    print(f"Logs saved to {log_dir}")
    print("Open TensorBoard to visualize the logs.")
    print("Run: tensorboard --logdir logs")