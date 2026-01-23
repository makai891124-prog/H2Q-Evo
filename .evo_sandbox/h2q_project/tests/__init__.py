import unittest

# Discover and load all test modules
loader = unittest.TestLoader()
start_dir = '.'  # Assuming tests are in the current directory or subdirectories
suite = loader.discover(start_dir)

# Create a test runner and run the tests
runner = unittest.TextTestRunner()
runner.run(suite)