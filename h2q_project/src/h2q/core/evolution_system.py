import os
import ast
from typing import List, Dict

from .loader import LocalFileLoader

class EvolutionSystem:
    def __init__(self, project_path: str, llm):
        self.project_path = project_path
        self.llm = llm
        self.file_loader = LocalFileLoader(project_path)

    def get_relevant_files(self, query: str) -> List[str]:
        # Placeholder implementation.  Ideally, this would use a more sophisticated
        # method like semantic search to identify relevant files.
        all_files = self.file_loader.load_all_files()
        return list(all_files.keys())

    def _extract_definitions(self, file_path: str, max_lines: int = 500) -> str:
        """Extract class and function definitions from a file, limiting line count.
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            return "" # or raise the exception if you prefer

        tree = ast.parse(content)
        definitions = []
        line_count = 0

        for node in tree.body:
            if line_count >= max_lines:
                break

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Extract the source code of the definition
                definition_str = ast.get_source_segment(content, node)
                if definition_str:
                    definitions.append(definition_str)
                    line_count += definition_str.count("\n") + 1 # Accurate line count

        return "\n".join(definitions)


    def get_project_context(self, query: str) -> str:
        relevant_files = self.get_relevant_files(query)
        context = ""
        for file_path in relevant_files:
            absolute_path = os.path.join(self.project_path, file_path)
            context += f"\n\n-- File: {file_path} --\n"
            context += self._extract_definitions(absolute_path)

        return context