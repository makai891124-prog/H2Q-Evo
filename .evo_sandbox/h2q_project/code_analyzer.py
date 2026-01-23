import ast
import os
import subprocess


class CodeAnalyzer:
    def __init__(self, project_root):
        self.project_root = project_root

    def analyze_file(self, file_path):
        issues = []

        # Check code style using pylint
        pylint_result = self.run_pylint(file_path)
        issues.extend(pylint_result)

        # Check type annotations using mypy
        mypy_result = self.run_mypy(file_path)
        issues.extend(mypy_result)

        # Check for potential numerical issues (basic example)
        numerical_issues = self.check_numerical_issues(file_path)
        issues.extend(numerical_issues)

        return issues

    def run_pylint(self, file_path):
        try:
            command = ["pylint", file_path]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            output = result.stdout
            errors = []
            for line in output.splitlines():
                if ":E" in line:
                    errors.append(f"Pylint: {line}")
            return errors
        except subprocess.CalledProcessError as e:
            return [f"Pylint failed: {e}"]
        except FileNotFoundError:
            return ["Pylint not found. Please install it."]

    def run_mypy(self, file_path):
        try:
            command = ["mypy", file_path]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            output = result.stdout
            errors = []
            for line in output.splitlines():
                if ": error:" in line:
                    errors.append(f"Mypy: {line}")
            return errors
        except subprocess.CalledProcessError as e:
            return [f"Mypy failed: {e}"]
        except FileNotFoundError:
            return ["Mypy not found. Please install it."]

    def check_numerical_issues(self, file_path):
        issues = []
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, ast.Div):
                    try:
                        right_operand_value = ast.literal_eval(node.right)
                        if right_operand_value == 0:
                            issues.append(f"Potential division by zero at line {node.lineno}")
                    except (ValueError, TypeError, AttributeError, SyntaxError):
                        # Ignore cases where the right operand is not a literal zero
                        pass

        return issues

    def analyze_project(self):
        all_issues = {}
        for root, _, files in os.walk(self.project_root):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.project_root)
                    issues = self.analyze_file(file_path)
                    if issues:
                        all_issues[relative_path] = issues
        return all_issues


if __name__ == '__main__':
    # Example Usage:
    project_root = "."  # Replace with your project root
    analyzer = CodeAnalyzer(project_root)
    all_issues = analyzer.analyze_project()

    if all_issues:
        print("Code analysis issues found:")
        for file, issues in all_issues.items():
            print(f"File: {file}")
            for issue in issues:
                print(f"  - {issue}")
    else:
        print("No code analysis issues found.")