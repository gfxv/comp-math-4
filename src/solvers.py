import math
import matplotlib.pyplot as plt
import numpy as np


########################
# LINEAR APPROXIMATION #
########################

def linear(x_points: list[float], y_points: list[float]) -> None:
    linear_y_points = linear_approximation(x_points, y_points)
    epsilon = [round(phi - y, 3) for phi, y in zip(linear_y_points, y_points)]
    S = round(sum(list(map(lambda e: e*e, epsilon))), 3)
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
    plt.show()


def linear_approximation(x_points: list[float], y_points: list[float]) -> list[float]:
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

    return [round(linear_result_function(a, b, x), 3) for x in x_points]


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
    S = round(sum(list(map(lambda e: e*e, epsilon))), 3)

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
    plt.show()


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
    plt.show()


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