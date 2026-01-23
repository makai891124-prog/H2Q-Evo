from h2q_project.dependency_analyzer import DependencyAnalyzer
import os

class CycleBreaker:
    def __init__(self, project_root, dependency_analyzer=None):
        self.project_root = project_root
        if dependency_analyzer is None:
            self.dependency_analyzer = DependencyAnalyzer(project_root)
            self.dependency_analyzer.analyze_dependencies() # Ensure dependencies are analyzed
        else:
            self.dependency_analyzer = dependency_analyzer
        self.cycles = self.dependency_analyzer.detect_cycles()

    def break_cycles(self):
        if not self.cycles:
            print("No cycles detected.")
            return

        for cycle in self.cycles:
            self.break_cycle(cycle)

    def break_cycle(self, cycle):
        """Breaks a specific cycle by removing the least important dependency."""

        #Find edge to remove.  For now, remove first dependency. Improve later.
        module1 = cycle[0]
        module2 = cycle[1]

        self.remove_dependency(module1, module2)


    def remove_dependency(self, module1, module2):
        """Removes the dependency of module2 from module1 by modifying the source file."""
        file_path1 = self.find_file_path(module1)

        if not file_path1:
            print(f"Warning: Could not find file for module {module1}")
            return

        with open(file_path1, 'r') as f:
            lines = f.readlines()

        modified_lines = []
        dependency_removed = False

        with open(file_path1, 'w') as f:
            tree = self.analyze_file(file_path1)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == module2:
                            #Remove entire line
                            lineno = node.lineno
                            for i in range(len(lines)):
                                if i+1 == lineno:
                                    lines[i] = ''
                                    dependency_removed = True
                                    break
                elif isinstance(node, ast.ImportFrom):
                    if node.module == module2:
                         #Remove entire line
                        lineno = node.lineno
                        for i in range(len(lines)):
                            if i+1 == lineno:
                                lines[i] = ''
                                dependency_removed = True
                                break
            f.writelines(lines)


        if dependency_removed:
            print(f"Removed dependency of {module2} from {module1} in {file_path1}")
            #Reanalyze dependencies after modification.
            self.dependency_analyzer.analyze_dependencies()
            self.cycles = self.dependency_analyzer.detect_cycles()
        else:
            print(f"Warning: Could not find dependency of {module2} in {module1} to remove.")

    def find_file_path(self, module_name):
        """Finds the file path for a given module name."""
        module_path = module_name.replace('.', os.sep) + ".py"
        full_path = os.path.join(self.project_root, module_path)

        if os.path.exists(full_path):
            return full_path
        else:
            return None

    def analyze_file(self, file_path):
        import ast
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
        return tree

import ast
