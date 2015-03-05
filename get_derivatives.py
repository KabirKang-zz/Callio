"""
get_derivatives.py
"""

def get_first_derivative(data_points):
    """
    Estimates a numerical first derivative.

    data_points should be an array like [[x, f(x)], [x+h, f(x+h)]].
    This returns an estimate of df / dx using a first order backward difference formula.
    The data points need to have -, +, and * available.
    """

    if len(data_points) != 2:
        raise RuntimeError()

    x_0 = data_points[0][0]
    f_0 = data_points[0][1]

    x_1 = data_points[1][0]
    f_1 = data_points[1][1]
    h = x_1 - x_0
    if h <= 0:
        raise ArithmeticError()
    return (f_1 - f_0) / h

def get_second_derivative(data_points):
    """
    Estimates a numerical second derivative.

    data_points should be an array like [[x, f(x)], [x+h, f(x+h)], [x + h + h', f(x + h + h')]].
    This returns an estimate of d^2f / dx^2 using a second order backward difference formula
    that can handle non-uniform grid spacing.
    The data points need to have -, +, and * available.
    """

    if len(data_points) != 3:
        raise RuntimeError()

    x_0 = data_points[0][0]
    f_0 = data_points[0][1]

    x_1 = data_points[1][0]
    f_1 = data_points[1][1]

    h = x_1 - x_0
    if h <= 0:
        raise ArithmeticError()

    x_2 = data_points[2][0]
    f_2 = data_points[2][1]

    alpha = (x_2 - x_1) / h
    if alpha <= 0:
        raise ArithmeticError()

    numerator = f_2 - f_1 * (1 + alpha) + alpha * f_0
    denomerator = alpha * (1 + alpha) * h * h
    return 2 * numerator / denomerator
