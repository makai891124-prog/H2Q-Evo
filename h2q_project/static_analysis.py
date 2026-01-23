import subprocess
import os

def run_pylint(file_path):
    try:
        result = subprocess.run(['pylint', file_path], capture_output=True, text=True, check=True)
        print(f"Pylint report for {file_path}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Pylint failed for {file_path}:\n{e.stderr}")

def run_mypy(file_path):
    try:
        result = subprocess.run(['mypy', file_path], capture_output=True, text=True, check=True)
        print(f"Mypy report for {file_path}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Mypy failed for {file_path}:\n{e.stderr}")


if __name__ == '__main__':
    # Example Usage
    current_dir = os.path.dirname(os.path.abspath(__file__))
    quaternion_file = os.path.join(current_dir, 'quaternion.py')
    test_quaternion_file = os.path.join(current_dir, 'test_quaternion.py')

    run_pylint(quaternion_file)
    run_pylint(test_quaternion_file)
    run_mypy(quaternion_file)
    run_mypy(test_quaternion_file)