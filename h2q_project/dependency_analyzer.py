import ast
import os

class DependencyAnalyzer:
    def __init__(self, project_root):
        self.project_root = project_root
        self.dependencies = {}
        self.reverse_dependencies = {}

    def analyze_dependencies(self):
        for root, _, files in os.walk(self.project_root):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    module_name = self.get_module_name(file_path)
                    self.dependencies[module_name] = self.get_imported_modules(file_path)

        # Build reverse dependencies after forward dependencies are complete
        self.build_reverse_dependencies()

    def get_module_name(self, file_path):
        relative_path = os.path.relpath(file_path, self.project_root)
        module_name = relative_path[:-3].replace(os.sep, '.')  # Remove .py and replace separators with dots
        return module_name

    def get_imported_modules(self, file_path):
        imported_modules = set()
        with open(file_path, 'r') as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError as e:
                print(f"SyntaxError in {file_path}: {e}")
                return imported_modules # Return empty set in case of error

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_modules.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module
                    if module:
                        imported_modules.add(module)
        return imported_modules

    def build_reverse_dependencies(self):
        for module, dependencies in self.dependencies.items():
            for dependency in dependencies:
                if dependency not in self.reverse_dependencies:
                    self.reverse_dependencies[dependency] = set()
                self.reverse_dependencies[dependency].add(module)

    def detect_cycles(self):
        visited = set()
        recursion_stack = set()
        cycles = []

        def dfs(module, path):
            visited.add(module)
            recursion_stack.add(module)

            for neighbor in self.dependencies.get(module, []):
                if neighbor in recursion_stack:
                    cycle = path + [neighbor]
                    cycles.append(cycle)
                    return True  # Cycle detected, stop exploring this branch
                elif neighbor not in visited:
                    if dfs(neighbor, path + [neighbor]):
                        return True  # Cycle detected in a deeper branch

            recursion_stack.remove(module)
            return False

        for module in self.dependencies:
            if module not in visited:
                dfs(module, [module])

        return cycles

    def get_all_dependencies(self, module):
        """Gets all dependencies (direct and transitive) for a given module."""
        all_deps = set()
        visited = set()

        def dfs(curr_module):
            visited.add(curr_module)
            direct_deps = self.dependencies.get(curr_module, [])
            for dep in direct_deps:
                if dep not in visited:
                    all_deps.add(dep)
                    dfs(dep)
        dfs(module)
        return all_deps
