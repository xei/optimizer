from continuous_optimization.gradient_descent import gradient_descent


def f(x):
    y = (x[0] ** 2 + (x[1] - 1) ** 2 + x[2] ** 2) + 2
    # y = (x[0] ** 2 + x[1] ** 2) / 2
    # y = (x[0] + 1) ** 2 + 1
    return y


def domain():
    d = [(-100, 100), (-100, 100), (-100, 100)]
    # d = [(-100, 100), (-100, 100)]
    # d = [(-100, 100)]
    return d


def grad_f(x):
    grad = [2 * x[0], 2 * (x[1] - 1), 2 * x[2]]
    # grad = [x[0], x[1]]
    # grad = [2 * (x[0] + 1)]
    return grad


def main():
    precision = 0.0001
    (optimal_point, optimal_value) = gradient_descent(f, domain(), grad_f, precision)
    print(optimal_point, optimal_value)


main()
