import os
import logging

logger = logging.getLogger(__name__)


def get_project_context(project_root):
    """Collects context information from the project.

    Args:
        project_root (str): The root directory of the project.

    Returns:
        dict: A dictionary containing project context information.
              Returns an empty dictionary if any error occurs.
    """
    context = {}
    context['project_root'] = project_root

    # Example: Read the contents of a requirements.txt file
    requirements_path = os.path.join(project_root, 'requirements.txt')
    try:
        with open(requirements_path, 'r') as f:
            context['requirements'] = f.read()
    except FileNotFoundError:
        logger.warning(f'Requirements file not found at {requirements_path}')
        context['requirements'] = None
    except Exception as e:
        logger.error(f'Error reading requirements file: {e}')
        context['requirements'] = None

    # Example: Read the contents of a pyproject.toml file if it exists
    pyproject_path = os.path.join(project_root, 'pyproject.toml')
    try:
        with open(pyproject_path, 'r') as f:
            context['pyproject'] = f.read()
    except FileNotFoundError:
        logger.warning(f'pyproject.toml file not found at {pyproject_path}')
        context['pyproject'] = None
    except Exception as e:
        logger.error(f'Error reading pyproject.toml file: {e}')
        context['pyproject'] = None

    # Add more context gathering logic here as needed

    return context
