# This is a sample Python file.

# Function to add two numbers
def add(x, y):
    """Add two numbers together.

    This function takes two numbers as input and returns their sum.

    Args:
        x: The first number.
        y: The second number.

    Returns:
        The sum of x and y.
    """
    return x + y


# Function to subtract two numbers
def subtract(x, y):
    """Subtract two numbers.

    This function takes two numbers as input and returns their difference (x - y).

    Args:
        x: The first number.
        y: The second number.

    Returns:
        The difference of x and y.
    """
    return x - y


# Main function to demonstrate the usage of add and subtract
def main():
    """Demonstrates the usage of the add and subtract functions.

    This function calls the add and subtract functions with sample values and prints the results.
    """
    num1 = 10
    num2 = 5

    sum_result = add(num1, num2)
    print(f"The sum of {num1} and {num2} is: {sum_result}")

    difference_result = subtract(num1, num2)
    print(f"The difference of {num1} and {num2} is: {difference_result}")


# Entry point of the script
if __name__ == "__main__":
    main()
