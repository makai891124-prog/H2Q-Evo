import json
import logging
from h2q_project.utils import handle_error, ConfigurationError

logger = logging.getLogger(__name__)


def load_config(config_path):
    """Loads configuration from a JSON file.

    Handles potential file not found or JSON decoding errors.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        handle_error(f"Configuration file not found: {config_path}")
        raise ConfigurationError(f"Configuration file not found: {config_path}") from None
    except json.JSONDecodeError as e:
        handle_error(f"Error decoding JSON in {config_path}", e)
        raise ConfigurationError(f"Error decoding JSON in {config_path}") from None
    except Exception as e:
        handle_error(f"Unexpected error loading configuration from {config_path}", e)
        raise ConfigurationError(f"Unexpected error loading configuration from {config_path}") from None
