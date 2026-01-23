import logging
from h2q_project.utils import safe_operation, handle_error, ValidationError

logger = logging.getLogger(__name__)


def process_data(data):
    """Processes the input data.

    Includes error handling for data validation.
    """
    if not isinstance(data, list):
        handle_error("Invalid data format: Data must be a list.")
        return None

    processed_data = []
    for item in data:
        try:
            validated_item = validate_item(item)
            processed_item = transform_item(validated_item)
            processed_data.append(processed_item)
        except ValidationError as e:
            handle_error(f"Validation error for item {item}", e)
        except Exception as e:
            handle_error(f"Error processing item {item}", e)

    return processed_data



def validate_item(item):
    """Validates a single data item.

    Raises ValidationError if validation fails.
    """
    if not isinstance(item, dict):
        raise ValidationError("Item must be a dictionary.")
    if 'id' not in item or 'value' not in item:
        raise ValidationError("Item must contain 'id' and 'value' keys.")
    return item



def transform_item(item):
    """Transforms a single data item.

    Performs a simple transformation.
    """
    item['value'] = item['value'] * 2
    return item
