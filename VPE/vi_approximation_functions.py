import numpy as np


###################################################################################################
# Individual empiric functions that can be fitted to the data

def linear(t, a, b):
    return a + b * t


def quadratic(t, a, b, c):
    return a + b * np.power(c * t, 2)


def cubic(t, a, b, c, n):
    return a + b * np.power(c * t, 3)


def polynomial(t, a, b, c, n):
    return a + b * np.power(c * t, n)


def exp(t, a, b, c):
    return a + b * np.exp(c * t)


def exp2(t, a, b, c):
    return a + b * np.power(np.exp(c * t), 2)


def exp3(t, a, b, c):
    return a + b * np.power(np.exp(c * t), 3)


def expn(t, a, b, c, n):
    return a + b * np.power(np.exp(c * t), 3 * n)


def log(t, a, b, c):
    return a + b * np.log(c * t)


def log2(t, a, b, c):
    return a + b * np.power(np.log(c * t), 2)


def log3(t, a, b, c):
    return a + b * np.power(np.log(c * t), 3)


def logn(t, a, b, c, n):
    return a + b * np.power(np.log(c * t), 3 * n)


###################################################################################################

FUNCTION_MAPPINGS = {"Linear": linear, "Quadratic": quadratic, "Cubic": cubic, "Polynomial": polynomial,
                     "Exponential^2": exp2, "Exponential^3": exp3,
                     "Exponential^n": expn, "Exponential": exp, "Logarithm^n": logn, "Logarithm": log,
                     "Logarithm^2": log2, "Logarithm^3": log3,
                     "Default": polynomial, "empiric_function": polynomial, '': polynomial}
