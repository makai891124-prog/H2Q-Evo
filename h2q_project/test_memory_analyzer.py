import unittest
import os
from h2q_project.memory_analyzer import MemoryAnalyzer  # Corrected import path

class TestMemoryAnalyzer(unittest.TestCase):

    def setUp(self):
        # Create a dummy project directory and files for testing
        self.project_root = 'test_project'
        os.makedirs(self.project_root, exist_ok=True)
        self.file1_path = os.path.join(self.project_root, 'file1.py')
        self.file2_path = os.path.join(self.project_root, 'file2.py')

        with open(self.file1_path, 'w') as f:
            f.write("data = []  # A potentially large list\n" +
                    "for i in range(1000): data.append(i)\n" +
                    "import pandas as pd\n"+
                    "df = pd.read_csv('large_file.csv')") #Simulate large csv file reading

        with open(self.file2_path, 'w') as f:
            f.write("data = {}\n" +
                    "for i in range(1000): data[i] = i\n" +
                    "import numpy as np\n"+
                    "array = np.load('large_array.npy')") #Simulate large numpy array loading

        self.analyzer = MemoryAnalyzer(self.project_root)

    def tearDown(self):
        # Remove the dummy project directory and files
        import shutil
        shutil.rmtree(self.project_root)

    def test_analyze_file_finds_large_list(self):
        result = self.analyzer.analyze_file('file1.py')
        self.assertIsNotNone(result)
        self.assertTrue(result['potential_bottlenecks'])
        self.assertTrue(any(['large_file.csv' in item['line_content'] for item in result['large_data_structures']]))


    def test_analyze_file_finds_large_dict(self):
        result = self.analyzer.analyze_file('file2.py')
        self.assertIsNotNone(result)
        self.assertTrue(result['potential_bottlenecks'])
        self.assertTrue(any(['large_array.npy' in item['line_content'] for item in result['large_data_structures']]))

    def test_analyze_project(self):
        results = self.analyzer.analyze_project()
        self.assertEqual(len(results), 2)
        files_with_bottlenecks = [r['file'] for r in results if r['potential_bottlenecks']]
        self.assertEqual(len(files_with_bottlenecks),2)


if __name__ == '__main__':
    unittest.main()