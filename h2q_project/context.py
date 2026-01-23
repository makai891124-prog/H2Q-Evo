import os

def get_project_context(file_path):
    """Reads a file and returns its content along with the file path.

    Args:
        file_path (str): The relative path to the file within the h2q_project directory.

    Returns:
        dict: A dictionary containing the file path and content.
              Returns None if the file is too large or cannot be read.
    """
    max_token_limit = 4000  # Define a token limit (adjust as needed)
    try:
        # Construct the absolute file path (important for security)
        absolute_file_path = os.path.abspath(file_path)

        # Check if the file path is within the allowed directory
        if not absolute_file_path.startswith(os.path.abspath("h2q_project")):
            print(f"Error: File path '{file_path}' is outside the allowed directory.")
            return None

        file_size = os.path.getsize(absolute_file_path)
        # Approximate token count: 1 token ~ 4 characters
        if file_size / 4 > max_token_limit:
            print(f"Error: File '{file_path}' exceeds the token limit.")
            return None

        with open(absolute_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return {"file_path": file_path, "content": content}
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return None
