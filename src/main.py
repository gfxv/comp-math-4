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


def read_file(path: str) -> tuple:
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        x_points = list(map(lambda x: float(x.strip()), lines[0].split()))
        y_points = list(map(lambda y: float(y.strip()), lines[1].split()))

        return x_points, y_points


def f(x: float) -> float:
    return 2*x**4


def main(args) -> None:

    x_points = None
    y_points = None

    if len(args) > 1:
        x_points, y_points = read_file(args[1])
        if len(x_points) != len(y_points):
            print(f"Invalid number of points. (x: {len(x_points)}, y: {len(y_points)})")
            sys.exit(0)
    else:      
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
    main(sys.argv)
