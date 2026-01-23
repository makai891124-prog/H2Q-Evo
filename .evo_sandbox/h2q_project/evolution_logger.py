import json

class EvolutionLogger:
    def __init__(self, log_file='evolution.log'):
        self.log_file = log_file

    def log_evolution_step(self, input_data, output_data, result):
        log_entry = {
            'input': input_data,
            'output': output_data,
            'result': result
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def load_logs(self):
        logs = []
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    logs.append(json.loads(line))
        except FileNotFoundError:
            pass
        return logs