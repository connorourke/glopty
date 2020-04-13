import numpy as np


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    """
    x: vector of input values
    """

    n = len(x)
    s1 = sum(np.power(x, 2))
    s2 = sum(np.cos(c * x))
    ack = -a * np.exp(-b * np.sqrt(s1 / n)) - np.exp(s2 / n) + a + np.exp(1)
    return ack


def rastrigin(x):
    a = 10
    d = len(x)
    s = np.power(x, 2) - a * np.cos(2 * np.pi * x)
    return a * d + sum(s)  # + 5.0


def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
