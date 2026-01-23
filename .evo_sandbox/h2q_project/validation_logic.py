def validate_input(data):
    """Validates the input data."""
    if not isinstance(data, dict):
        raise TypeError("Input must be a dictionary")

    if 'name' not in data or not isinstance(data['name'], str) or not data['name']:
        raise ValueError("Name must be a non-empty string")

    if 'age' not in data or not isinstance(data['age'], int) or data['age'] <= 0:
        raise ValueError("Age must be a positive integer")

    if 'email' not in data or not isinstance(data['email'], str) or '@' not in data['email']:
        raise ValueError("Email must be a valid email address")

    return True
