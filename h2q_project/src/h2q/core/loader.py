"""
Local File Loader for H2Q Project
=================================

A simple utility to load and index all files in a project directory.
"""

import os
from pathlib import Path
from typing import Dict, Any


class LocalFileLoader:
    """
    Simple file loader that recursively scans a directory and loads file contents.
    """

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self._files_cache = None

    def load_all_files(self) -> Dict[str, str]:
        """
        Load all files in the project directory.

        Returns:
            Dict[str, str]: Dictionary mapping file paths to file contents
        """
        if self._files_cache is not None:
            return self._files_cache

        files = {}
        try:
            for root, dirs, filenames in os.walk(self.project_path):
                # Skip common directories that shouldn't be indexed
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]

                for filename in filenames:
                    # Skip binary files and common non-text files
                    if self._is_text_file(filename):
                        file_path = Path(root) / filename
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                rel_path = file_path.relative_to(self.project_path)
                                files[str(rel_path)] = content
                        except Exception:
                            # Skip files that can't be read
                            continue
        except Exception as e:
            print(f"Error loading files: {e}")

        self._files_cache = files
        return files

    def _is_text_file(self, filename: str) -> bool:
        """
        Check if a file is likely a text file based on extension.
        """
        text_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
            '.txt', '.md', '.rst', '.json', '.yaml', '.yml', '.xml',
            '.html', '.css', '.scss', '.sh', '.bat', '.ps1', '.sql',
            '.r', '.m', '.go', '.rs', '.php', '.rb', '.pl', '.lua'
        }

        ext = Path(filename).suffix.lower()
        return ext in text_extensions or not ext  # Files without extension might be text