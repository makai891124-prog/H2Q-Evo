import psutil
import os
import json
import time

class ContainerProfiler:
    def __init__(self, container_name="h2q_container", output_file="profiling_data.json", interval=1):
        self.container_name = container_name
        self.output_file = output_file
        self.interval = interval
        self.container_pid = self._get_container_pid()
        self.data = []

    def _get_container_pid(self):
        try:
            # Execute docker inspect command to get the container's PID
            command = f"docker inspect -f '{{{{.State.Pid}}}}' {self.container_name}"
            process = os.popen(command)
            pid_str = process.read().strip()
            process.close()
            return int(pid_str)
        except Exception as e:
            print(f"Error getting container PID: {e}")
            return None

    def _get_process_info(self):
        if not self.container_pid:
            return None

        try:
            process = psutil.Process(self.container_pid)
            cpu_percent = process.cpu_percent(interval=0.1) # Short interval to reduce overall impact
            memory_info = process.memory_info()
            return {
                "cpu_percent": cpu_percent,
                "memory_rss": memory_info.rss,  # Resident Set Size
                "memory_vms": memory_info.vms   # Virtual Memory Size
            }
        except psutil.NoSuchProcess:
            print("Process not found.")
            return None
        except Exception as e:
            print(f"Error getting process info: {e}")
            return None

    def run_profiling(self):
        if not self.container_pid:
            print("Container PID not found. Profiling cannot start.")
            return

        try:
            while True:
                process_info = self._get_process_info()
                if process_info:
                    timestamp = time.time()
                    self.data.append({"timestamp": timestamp, **process_info})
                    print(f"CPU: {process_info['cpu_percent']}%, RSS: {process_info['memory_rss']} bytes, VMS: {process_info['memory_vms']} bytes")

                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("Profiling stopped.")
            self.save_data()
        except Exception as e:
            print(f"Profiling failed: {e}")

    def save_data(self):
        try:
            with open(self.output_file, 'w') as f:
                json.dump(self.data, f, indent=4)
            print(f"Profiling data saved to {self.output_file}")
        except Exception as e:
            print(f"Error saving data to file: {e}")

if __name__ == '__main__':
    profiler = ContainerProfiler()
    profiler.run_profiling()
