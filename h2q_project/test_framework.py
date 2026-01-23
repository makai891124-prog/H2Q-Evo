import unittest
import os
import subprocess
import sys

class BaseTestCase(unittest.TestCase):
    """Base class for all test cases. Provides common setup and teardown.
       Designed to be lightweight and Docker-friendly.
    """

    def setUp(self):
        """Setup method executed before each test.
           Override this in subclasses to add specific setup logic.
        """
        self.env = os.environ.copy()
        self.project_root = os.path.dirname(os.path.abspath(__file__))

    def tearDown(self):  # Corrected method name
        """Teardown method executed after each test.
           Override this in subclasses to add specific teardown logic.
        """
        pass

    def run_command(self, command, env=None, check=True, capture_output=True, text=True):
        """Helper function to run a shell command.
           Simplified for Docker environment: relies on subprocess directly.
        """
        effective_env = self.env.copy()
        if env:
            effective_env.update(env)

        process = subprocess.run(command, capture_output=capture_output, text=text, env=effective_env)

        if check:
            process.check_returncode()

        return process

    def assertCommandSuccess(self, process):
        """Assert that a command executed successfully.
        """
        self.assertEqual(process.returncode, 0, f"Command failed with return code {process.returncode}: {process.stderr}")

    def assertCommandFailure(self, process):
        """Assert that a command executed with a non-zero return code.
        """
        self.assertNotEqual(process.returncode, 0, "Command unexpectedly succeeded")


if __name__ == '__main__':
    unittest.main()
