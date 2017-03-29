import random

MAX_UNDULY_ITERATIONS = 10


def get_step_size(i):
    """
    This function return the step-size (gamma) of the Gradient-descent algorithm based on the iteration number.
    As the algorithm performs, the step-size will decrease.
    :param i: The iteration number
    :return: The step-size
    """
    gamma = 1 / i
    return gamma


def gradient_descent(f, domain, grad_f, precision):
    """
    The Gradient-descent algorithm is a first-order iterative algorithm inspired by Euler method in ODEs that
    finds a local minimum of a differentiable function.
    If the function be convex, the local minimum found by the Gradient-descent is the global minimum.

    :param f: The function that is considered for minimization. This must be a differentiable real function of
    one/several variables. For finding the global minimum, this must be a convex function (i.e. Quadratic function).
    example:
        def f(x):
            y = (x[0] ** 2 + x[1] ** 2) / 2
            return y
    :param domain: This list present the domain of any variables in the function. The algorithm starts from a random
    point in this space and try to find a better point.
    example:
        [
            (-100, 100),
            (-100, 100)
        ]
    :param grad_f: This is the Gradient of a multi-variable function or the Derivative of a function of one variable.
    Calculating it from the function can be time-consuming so the invoker must calculate and pass it.
    Example:
        def grad_f(x):
            return [x[0], x[1]]
    :param precision: The optimization process terminate when the function value changes reach this precision.
    Example:
        0.0001
    :return: A tuple include the minimum point and it's value.
    """

    x = [random.uniform(d[0], d[1]) for d in domain]

    unduly_iterations_counter = 0
    iteration = 1
    while unduly_iterations_counter < MAX_UNDULY_ITERATIONS:

        x_last = x
        grad = grad_f(x)
        step_size = get_step_size(iteration)
        x = [x_i - step_size * grad_i for x_i, grad_i in zip(x, grad)]

        if abs(f(x) - f(x_last)) > precision:
            unduly_iterations_counter = 0
        else:
            unduly_iterations_counter += 1

        iteration += 1

    return x, f(x)
