import subprocess
import sys
import os

class CodeStyleChecker:
    def __init__(self, target_directory='h2q_project', linter='flake8'):
        self.target_directory = target_directory
        self.linter = linter

    def run_linter(self):
        try:
            command = [self.linter, self.target_directory]
            result = subprocess.run(command, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                print(f"Code style check failed with {self.linter}:")
                print(result.stdout)
                print(result.stderr)
                return False
            else:
                print(f"Code style check passed with {self.linter}.")
                return True

        except FileNotFoundError:
            print(f"Error: {self.linter} not found. Please ensure it is installed.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False

    def check_all_files(self):
      # Check only python files in the target_directory
      python_files = []
      for root, _, files in os.walk(self.target_directory):
        for file in files:
          if file.endswith(".py"):
            python_files.append(os.path.join(root, file))

      all_passed = True # Flag to track success
      for py_file in python_files:
        try:
          command = [self.linter, py_file]
          result = subprocess.run(command, capture_output=True, text=True, check=False)

          if result.returncode != 0:
            print(f"Code style check failed with {self.linter} on {py_file}:")
            print(result.stdout)
            print(result.stderr)
            all_passed = False  # Set failure flag
          else:
            print(f"Code style check passed with {self.linter} on {py_file}.")

        except FileNotFoundError:
          print(f"Error: {self.linter} not found. Please ensure it is installed.")
          return False

        except Exception as e:
          print(f"An unexpected error occurred: {e}")
          all_passed = False  # Treat exceptions as style failures

      return all_passed


if __name__ == '__main__':
    checker = CodeStyleChecker()
    if not checker.check_all_files():
        sys.exit(1)  # Exit with an error code if checks fail
    else:
        sys.exit(0)
