import time
from h2q_project.performance_monitor import PerformanceMonitor

def train_model(epochs=5):
    monitor = PerformanceMonitor(interval=1)
    print("Starting training...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        # Simulate training workload
        time.sleep(2) 
        monitor.record()
    print("Training complete.")
    monitor.save_to_json("training_performance.json")

if __name__ == '__main__':
    train_model()