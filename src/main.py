import sys

from solvers import linear, quadractic, cubic, power, exponential, logarithmic

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


    print()
    linear(x_points, y_points)
    print()
    quadractic(x_points, y_points)
    print()
    cubic(x_points, y_points)
    
    try:
        print()
        power(x_points, y_points)
    except ValueError as e:
        print(e)

    try:
        print()
        exponential(x_points, y_points)
    except ValueError as e:
        print(e)

    try:
        print()
        logarithmic(x_points, y_points)
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()