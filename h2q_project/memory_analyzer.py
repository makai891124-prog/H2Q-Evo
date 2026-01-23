import gc
import tracemalloc
import json
import os

class MemoryAnalyzer:
    def __init__(self):
        self.snapshots = []

    def start_tracking(self):
        tracemalloc.start()

    def take_snapshot(self, description):
        self.snapshots.append({
            'description': description,
            'snapshot': tracemalloc.take_snapshot()
        })

    def stop_tracking(self):
        tracemalloc.stop()

    def generate_report(self, output_path='memory_report.json'):
        report = []
        for i in range(len(self.snapshots) - 1):
            snapshot1 = self.snapshots[i]['snapshot']
            snapshot2 = self.snapshots[i+1]['snapshot']
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')

            snapshot_data = {
                'from': self.snapshots[i]['description'],
                'to': self.snapshots[i+1]['description'],
                'leaks': []
            }

            for stat in top_stats[:10]: # Limit to top 10 leaks
                snapshot_data['leaks'].append({
                    'file': stat.traceback[0].filename,
                    'line': stat.traceback[0].lineno,
                    'size_diff': stat.size_diff,
                    'size': stat.size,
                    'count_diff': stat.count_diff,
                    'count': stat.count
                })

            report.append(snapshot_data)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)

        print(f'Memory analysis report generated at {output_path}')

    def run_garbage_collection(self):
        gc.collect()


if __name__ == '__main__':
    # Example Usage
    analyzer = MemoryAnalyzer()

    analyzer.start_tracking()

    # Simulate some memory allocation
    data = [i for i in range(1000000)]
    analyzer.take_snapshot('After initial allocation')

    # Simulate some deallocation (but with a potential leak)
    data = None
    analyzer.run_garbage_collection()
    analyzer.take_snapshot('After deallocation and GC')

    analyzer.stop_tracking()
    analyzer.generate_report()
