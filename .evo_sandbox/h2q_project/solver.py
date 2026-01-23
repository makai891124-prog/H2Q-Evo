import sympy

class Solver:
    def __init__(self, equation_string):
        self.equation_string = equation_string

    def solve(self):
        try:
            x = sympy.symbols('x')
            equation = sympy.sympify(self.equation_string)
            solutions = sympy.solve(equation, x)
            return solutions
        except (SyntaxError, TypeError, ValueError) as e:
            return f"Error: Invalid equation format. Please check your input. Details: {e}"
        except Exception as e:
            return f"Error: An unexpected error occurred: {e}"

if __name__ == '__main__':
    # Example Usage:
    equation_string = "x**2 - 4 = 0"  # Example equation
    solver = Solver(equation_string)
    solutions = solver.solve()
    print(f"Solutions for equation '{equation_string}': {solutions}")
