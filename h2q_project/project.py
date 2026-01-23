import os
import glob

class Project:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def get_all_code_files(self):
        return self._find_files_with_extension(self.root_dir, '.py')

    def _find_files_with_extension(self, path, extension):
        pattern = os.path.join(path, f'**/*{extension}')
        files = glob.glob(pattern, recursive=True)
        return [f for f in files if '__pycache__' not in f]

    def get_file_content(self, file_path):
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return None

    def get_project_context(self, current_task=None):
        # Prioritize files related to the current task, if available
        if current_task:
            relevant_files = []
            for file in self.get_all_code_files():
                if current_task.lower() in file.lower():
                    relevant_files.append(file)

            if relevant_files:
                return [{'file_path': os.path.relpath(f, self.root_dir), 'content': self.get_file_content(f)} for f in relevant_files]

        # Fallback to all code files if no relevant files are found or no task is provided
        all_files = self.get_all_code_files()
        return [{'file_path': os.path.relpath(f, self.root_dir), 'content': self.get_file_content(f)} for f in all_files]
