import cmath

def solve_equation(a, b, c):
  """Solves a quadratic equation of the form ax^2 + bx + c = 0."""
  delta = (b**2 - 4*a*c)
  if delta >= 0:
    x1 = (-b - delta**0.5) / (2*a)
    x2 = (-b + delta**0.5) / (2*a)
    return x1, x2
  else:
    x1 = (-b - cmath.sqrt(delta)) / (2 * a)
    x2 = (-b + cmath.sqrt(delta)) / (2 * a)
    return x1, x2