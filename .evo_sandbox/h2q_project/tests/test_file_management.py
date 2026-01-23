import unittest
import os
from h2q_project.file_manager import FileManager

class TestFileManager(unittest.TestCase):
    def setUp(self):
        self.file_manager = FileManager(base_dir='test_temp')
        # Clean up the test directory before each test
        if os.path.exists('test_temp'):
            import shutil
            shutil.rmtree('test_temp')
        os.makedirs('test_temp', exist_ok=True)

    def tearDown(self):
        # Clean up the test directory after each test
        if os.path.exists('test_temp'):
            import shutil
            shutil.rmtree('test_temp')

    def test_create_and_combine_chunks(self):
        data = [b'This is the first chunk.', b'This is the second chunk.', b'This is the third chunk.']
        base_filename = 'test_chunks'
        output_filename = 'combined_file.txt'

        filepath = self.file_manager.create_chunked_file(base_filename, data)
        self.assertIsNotNone(filepath)

        combined_path = self.file_manager.combine_chunks(base_filename, output_filename)
        self.assertIsNotNone(combined_path)

        with open(combined_path, 'rb') as f:
            combined_content = f.read()
        expected_content = b''.join(data)
        self.assertEqual(combined_content, expected_content)

    def test_write_small_file(self):
        filename = 'small_file.txt'
        content = 'This is a small file content.'
        filepath = self.file_manager.write_small_file(filename, content)
        self.assertIsNotNone(filepath)

        with open(filepath, 'r') as f:
            read_content = f.read()
        self.assertEqual(read_content, content)

if __name__ == '__main__':
    unittest.main()
