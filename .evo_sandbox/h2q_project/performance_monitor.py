class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}

    def track_metric(self, metric_name, value):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def get_metric(self, metric_name):
        return self.metrics.get(metric_name, [])
