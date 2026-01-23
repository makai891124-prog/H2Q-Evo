def multiply(x, y):
    """Multiplies two numbers."""
    return x * y


def divide(x, y):
    """Divides two numbers. Handles division by zero."""
    if y == 0:
        return "Cannot divide by zero!"
    return x / y