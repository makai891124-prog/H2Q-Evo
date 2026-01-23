import os

class FileManager:
    def __init__(self, base_dir='h2q_project_temp'):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def create_chunked_file(self, filename, data_iterable, chunk_size=1024*1024): # 1MB chunk size
        filepath = os.path.join(self.base_dir, filename)
        chunk_index = 0
        try:
            with open(filepath + f'.part{chunk_index:03d}', 'wb') as f:
                for data in data_iterable:
                    f.write(data)
        except Exception as e:
            print(f'Error writing to chunk: {e}')
            return None

        return filepath

    def combine_chunks(self, base_filename, output_filename):
        output_path = os.path.join(self.base_dir, output_filename)
        chunk_index = 0
        try:
            with open(output_path, 'wb') as outfile:
                while True:
                    chunk_filename = base_filename + f'.part{chunk_index:03d}'
                    chunk_filepath = os.path.join(self.base_dir, chunk_filename)
                    if not os.path.exists(chunk_filepath):
                        break
                    with open(chunk_filepath, 'rb') as infile:
                        outfile.write(infile.read())
                    os.remove(chunk_filepath)
                    chunk_index += 1

        except Exception as e:
             print(f'Error combining chunks: {e}')
             return None

        return output_path

    def write_small_file(self, filename, content):
        filepath = os.path.join(self.base_dir, filename)
        try:
            with open(filepath, 'w') as f:
                f.write(content)
            return filepath
        except Exception as e:
            print(f'Error writing small file: {e}')
            return None
