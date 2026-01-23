"""
Knot theory module.

This module provides functions for working with knots,
such as calculating their invariants and visualizing them.

"""

import numpy as np


def trefoil_polynomial(variable):
    """Calculates the Alexander polynomial of the trefoil knot.

    Args:
        variable (float): The variable to evaluate the polynomial at.

    Returns:
        float: The value of the Alexander polynomial at the given variable.
    """
    return variable**2 - variable + 1


def figure_eight_polynomial(variable):
    """Calculates the Alexander polynomial of the figure-eight knot.

    Args:
        variable (float): The variable to evaluate the polynomial at.

    Returns:
        float: The value of the Alexander polynomial at the given variable.
    """
    return variable**2 - 3*variable + 1


def knot_crossing_number(knot_type):
    """Returns the crossing number of a given knot type.

    Args:
        knot_type (str): The name of the knot type (e.g., 'trefoil', 'figure_eight').

    Returns:
        int: The crossing number of the knot.

    Raises:
        ValueError: If the knot type is not recognized.
    """
    if knot_type == 'trefoil':
        return 3
    elif knot_type == 'figure_eight':
        return 4
    else:
        raise ValueError("Unknown knot type: {}".format(knot_type))
