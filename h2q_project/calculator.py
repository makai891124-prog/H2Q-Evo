def add(x, y):
    """Adds two numbers together."""
    return x + y

def subtract(x, y):
    """Subtracts one number from another."""
    return x - y

def multiply(x, y):
    """Multiplies two numbers together."""
    return x * y

def divide(x, y):
    """Divides one number by another."""
    if y == 0:
        raise ValueError("Cannot divide by zero")
    return x / y

def power(x, y):
    """Raises a number to a power."""
    return x ** y

def square_root(x):
    """Calculates the square root of a number."""
    if x < 0:
        raise ValueError("Cannot calculate square root of a negative number")
    return x ** 0.5

