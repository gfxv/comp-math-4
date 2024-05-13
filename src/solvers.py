import math
import matplotlib.pyplot as plt
import numpy as np

import utils

########################
# LINEAR APPROXIMATION #
########################

def linear(x_points: list[float], y_points: list[float]) -> None:
    linear_y_points = linear_approximation(x_points, y_points)
    epsilon = [round(phi - y, 3) for phi, y in zip(linear_y_points, y_points)]
    S = sum(list(map(lambda e: e*e, epsilon)))
    r = pearson_coefficient(x_points, y_points)

    print("Linear approximation:")
    print("x:", x_points)
    print("y:", y_points)
    print("ф:", linear_y_points)
    print("e:", epsilon)
    print("S:", S)
    print("r:", r)

    fig = plt.figure()
    ax = plt.axes()
    
    ax.scatter(x_points, y_points)
    ax.plot(x_points, linear_y_points)
    plt.savefig("linear.png")


def linear_approximation(x_points: list[float], y_points: list[float]) -> list[float]:
    a, b = linear_coefficients(x_points, y_points)
    return [round(linear_result_function(a, b, x), 3) for x in x_points]


def linear_coefficients(x_points: list[float], y_points: list[float]) -> tuple[float]:
    sx = sum(x_points)
    sxx = sum(list(map(lambda x: x*x, x_points)))
    sy = sum(y_points)
    sxy = sum([x*y for x, y in zip(x_points, y_points)])

    n = len(x_points)

    delta = sxx * n - sx * sx
    delta1 = sxy * n - sx * sy
    delta2 = sxx * sy - sx * sxy

    a = round(delta1 / delta, 3)
    b = round(delta2 / delta, 3)

    return a, b


def linear_result_function(a: float, b: float, x: float) -> float:
    return a * x + b


def pearson_coefficient(x_points: list[float], y_points: list[float]) -> float:
    mean_x = sum(x_points) / len(x_points)
    mean_y = sum(y_points) / len(y_points)

    top = sum([(x - mean_x)*(y - mean_y) for x, y in zip(x_points, y_points)])
    bottom = math.sqrt(sum([(x - mean_x)**2 for x in x_points]) * sum([(y - mean_y)**2 for y in y_points]))

    return round(top / bottom, 2)


###########################
# QUADRATIC APPROXIMATION #
###########################

def quadractic(x_points: list[float], y_points: list[float]) -> None:
    quadractic_y_points = quadractic_approximation(x_points, y_points)
    epsilon = [round(phi - y, 3) for phi, y in zip(quadractic_y_points, y_points)]
    S = sum(list(map(lambda e: e*e, epsilon)))

    print("Quadractic approximation:")
    print("x:", x_points)
    print("y:", y_points)
    print("ф:", quadractic_y_points)
    print("e:", epsilon)
    print("S:", S)

    fig = plt.figure()
    ax = plt.axes()
    
    ax.scatter(x_points, y_points)
    ax.plot(x_points, quadractic_y_points)
    plt.savefig("quadratic.png")


def quadractic_approximation(x_points: list[float], y_points: list[float]) -> list[float]:
    x = sum(x_points)
    x2 = sum(list(map(lambda x: x*x, x_points)))
    x3 = sum(list(map(lambda x: x**3, x_points)))
    x4 = sum(list(map(lambda x: x**4, x_points)))
    y = sum(y_points)
    xy = sum([x * y for x, y in zip(x_points, y_points)])
    x2y = sum([x * x * y for x, y in zip(x_points, y_points)])
    n = len(x_points)


    a = np.array([
        [n, x, x2],
        [x, x2, x3],
        [x2, x3, x4]
    ])
    b = np.array([y, xy, x2y])
    a0, a1, a2 = np.linalg.solve(a, b)

    return [round(quadractic_result_function(a0, a1, a2, x), 3) for x in x_points]

def quadractic_result_function(a0: float, a1: float, a2: float, x: float) -> float:
    return a0 + a1 * x + a2 * x * x


#######################
# CUBIC APPROXIMATION #
#######################

def cubic(x_points: list[float], y_points: list[float]) -> None:
    cubic_y_points = cubic_approximation(x_points, y_points)
    epsilon = [round(phi - y, 3) for phi, y in zip(cubic_y_points, y_points)]
    S = sum(list(map(lambda e: e*e, epsilon)))

    print("Cubic approximation:")
    print("x:", x_points)
    print("y:", y_points)
    print("ф:", cubic_y_points)
    print("e:", epsilon)
    print("S:", S)

    fig = plt.figure()
    ax = plt.axes()
    
    ax.scatter(x_points, y_points)
    ax.plot(x_points, cubic_y_points)
    plt.savefig("cubic.png")


def cubic_approximation(x_points: list[float], y_points: list[float]) -> list[float]:
    x = sum(x_points)
    x2 = sum(list(map(lambda x: x*x, x_points)))
    x3 = sum(list(map(lambda x: x**3, x_points)))
    x4 = sum(list(map(lambda x: x**4, x_points)))
    x5 = sum(list(map(lambda x: x**5, x_points)))
    x6 = sum(list(map(lambda x: x**6, x_points)))
    y = sum(y_points)
    xy = sum([x * y for x, y in zip(x_points, y_points)])
    x2y = sum([x * x * y for x, y in zip(x_points, y_points)])
    x3y = sum([x * x * x * y for x, y in zip(x_points, y_points)])
    n = len(x_points)

    a = np.array([
        [n, x, x2, x3],
        [x, x2, x3, x4],
        [x2, x3, x4, x5],
        [x3, x4, x5, x6]
    ])
    b = np.array([y, xy, x2y, x3y])
    a0, a1, a2, a3 = np.linalg.solve(a, b)

    return [round(cubic_result_function(a0, a1, a2, a3, x), 3) for x in x_points]


def cubic_result_function(a0: float, a1: float, a2: float, a3: float, x) -> float:
    return a0 + a1 * x + a2 * x * x + a3 * x**3


#############################
# EXPONENTIAL APPROXIMATION #
#############################

def exponential(x_points: list[float], y_points: list[float]) -> None:
    exponential_y_points = exponential_approximation(x_points, y_points)
    epsilon = [round(phi - y, 3) for phi, y in zip(exponential_y_points, y_points)]
    S = sum(list(map(lambda e: e*e, epsilon)))

    print("Exponential approximation:")
    print("x:", x_points)
    print("y:", y_points)
    print("ф:", exponential_y_points)
    print("e:", epsilon)
    print("S:", S)

    fig = plt.figure()
    ax = plt.axes()
    
    ax.scatter(x_points, y_points)
    ax.plot(x_points, exponential_y_points)
    plt.savefig("exponential.png")


def exponential_approximation(x_points: list[float], y_points: list[float]) -> list[float]:
    if not utils.all_exp_safe(y_points):
        raise ValueError("Can't perform exponential approximation. There are some invalid `y` values")
    
    a, b = linear_coefficients(x_points, y_points)

    return [round(exponentioal_result_funtion(math.exp(a), b, x), 3) for x in x_points]


def exponentioal_result_funtion(a: float, b: float, x: float) -> float:
    return a * math.exp(b * x)


#######################
# POWER APPROXIMATION #
#######################

def power(x_points: list[float], y_points: list[float]) -> None:
    power_y_points = power_approximation(x_points, y_points)
    epsilon = [round(phi - y, 3) for phi, y in zip(power_y_points, y_points)]
    S = sum(list(map(lambda e: e*e, epsilon)))

    print("Power approximation:")
    print("x:", x_points)
    print("y:", y_points)
    print("ф:", power_y_points)
    print("e:", epsilon)
    print("S:", S)

    fig = plt.figure()
    ax = plt.axes()
    
    ax.scatter(x_points, y_points)
    ax.plot(x_points, power_y_points)
    plt.savefig("power2.png")


def power_approximation(x_points: list[float], y_points: list[float]) -> list[float]:
    # n = len(x_points)

    # b_top = n * sum([math.log(x) * math.log(y) for x, y in zip(x_points, y_points)]) - sum(list(map(lambda x: math.log(x), x_points))) * sum(list(map(lambda y: math.log(y), y_points)))
    # b_bottom = n * sum(list(map(lambda x: math.log(x)**2, x_points))) - sum(list(map(lambda x: math.log(x), x_points)))**2
    # b = b_top / b_bottom

    # a_pow = sum(list(map(lambda y: math.log(y), y_points))) / n - (b/n) * sum(list(map(lambda x: math.log(x), x_points)))
    # a = math.exp(a_pow)

    if not utils.all_exp_safe(x_points) or not utils.all_exp_safe(y_points):
        raise ValueError("Can't perform power approximation. There are some invalid `x` or `y` values")

    ln_x_points = utils.to_ln(x_points)
    ln_y_points = utils.to_ln(y_points)

    a, b = linear_coefficients(ln_x_points, ln_y_points)

    return [round(power_result_funtion(math.exp(a), b, x), 3) for x in x_points]


def power_result_funtion(a: float, b: float, x: float) -> float:
    return a * (x**b)


#############################
# LOGARITHMIC APPROXIMATION #
#############################

def logarithmic(x_points: list[float], y_points: list[float]) -> None:
    lop_y_points = logarithmic_approximation(x_points, y_points)
    epsilon = [round(phi - y, 3) for phi, y in zip(lop_y_points, y_points)]
    S = sum(list(map(lambda e: e*e, epsilon)))

    print("Logarithmic approximation:")
    print("x:", x_points)
    print("y:", y_points)
    print("ф:", lop_y_points)
    print("e:", epsilon)
    print("S:", S)

    fig = plt.figure()
    ax = plt.axes()
    
    ax.scatter(x_points, y_points)
    ax.plot(x_points, lop_y_points)
    plt.savefig("logarithmic.png")


def logarithmic_approximation(x_points: list[float], y_points: list[float]) -> list[float]:

    if not utils.all_exp_safe(x_points):
        raise ValueError("Can't perform logarithmic approximation. There are some invalid `x` values")

    a, b = linear_coefficients(x_points, y_points)

    return [round(logarithmic_result_function(a, b, x), 3) for x in x_points]


def logarithmic_result_function(a: float, b: float, x: float) -> float:
    return a * math.log(x) + b