import json

from jsonschema import validate, ValidationError


def validate_json(json_data, schema):
    try:
        validate(instance=json_data, schema=schema)
    except ValidationError as e:
        return False, str(e)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return False, f"An unexpected error occurred during validation: {str(e)}"
    return True, None


if __name__ == '__main__':
    # Example Usage
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
        },
        "required": ["name", "age"]
    }

    valid_json = {"name": "Alice", "age": 30}
    invalid_json = {"name": "Bob", "age": -5}
    malformed_json = '{"name": "Charlie", "age": ' # Incomplete JSON

    is_valid, error_message = validate_json(valid_json, schema)
    print(f"Valid JSON: {is_valid}, Error: {error_message}")

    is_valid, error_message = validate_json(invalid_json, schema)
    print(f"Invalid JSON: {is_valid}, Error: {error_message}")

    is_valid, error_message = validate_json(malformed_json, schema)
    print(f"Malformed JSON: {is_valid}, Error: {error_message}")
