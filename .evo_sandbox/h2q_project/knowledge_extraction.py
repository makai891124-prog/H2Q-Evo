import ast
import inspect


def get_project_context(project_path):
    """Extracts relevant context from the project, focusing on file structure and function definitions."""
    context = {
        "file_structure": get_file_structure(project_path),
        "function_definitions": get_function_definitions(project_path)
    }
    return context


def get_file_structure(project_path):
    """Returns a dictionary representing the file structure of the project."""
    file_structure = {}
    for filepath in get_all_python_files(project_path):
        with open(filepath, 'r') as f:
            file_structure[filepath] = f.read()
    return file_structure


def get_all_python_files(project_path):
    """Helper function to get all python files within a project directory."""
    import os
    python_files = []
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def get_function_definitions(project_path):
    """Extracts function definitions (name, arguments, return type) from all Python files in the project."""
    function_definitions = []
    for filepath in get_all_python_files(project_path):
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_name = node.name
                    args = [arg.arg for arg in node.args.args]
                    # Attempt to extract return type hint (Python 3.5+)
                    return_type = ast.unparse(node.returns) if node.returns else None

                    function_definitions.append({
                        "filepath": filepath,
                        "name": function_name,
                        "arguments": args,
                        "return_type": return_type
                    })
    return function_definitions
