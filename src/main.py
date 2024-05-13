import sys

from solvers import linear, quadractic, cubic, power, exponential, logarithmic
import utils

def read_input():
    print("Input boundaries:")
    try:
        a = int(input("A (left boundary): "))
        b = int(input("B (right boundary): "))
        h = float(input("Interval length: "))
    except ValueError as e:
        print("Invalid argument passed!", e)
        sys.exit(0)

    return min(a, b), max(a, b), h


def f(x: float) -> float:
    return 4*x / (x**4 + 12)


def main() -> None:
    a, b, h = read_input()
    interval_count = int((b - a) / h) + 1
    print("Interval count:", interval_count)
    x_points = [round(a + i * h, 3) for i in range(interval_count)]
    y_points = [round(f(x), 3) for x in x_points]
    print("Intervals: ", x_points)

    reliabilities = []

    print()
    reliabilities.append((linear(x_points, y_points), "linear"))
    print()
    reliabilities.append((quadractic(x_points, y_points), "quadractic"))
    print()
    reliabilities.append((cubic(x_points, y_points), "cubic"))

    try:
        print()
        reliabilities.append((power(x_points, y_points), "power"))
    except ValueError as e:
        print(e)

    try:
        print()
        reliabilities.append((exponential(x_points, y_points), "exponential"))
    except ValueError as e:
        print(e)

    try:
        print()
        reliabilities.append((logarithmic(x_points, y_points), "logarithmic"))
    except ValueError as e:
        print(e)

    m, name = utils.max_reliability(reliabilities)

    print()
    print("Reliabilities of approximation (coefficients of determination):")
    utils.print_reliabilities(reliabilities)
    print()
    print("Best method:", name)
    print("With reliability:", m)


if __name__ == "__main__":
    main()