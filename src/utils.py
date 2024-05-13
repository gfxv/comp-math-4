import math

def all_exp_safe(data: list[float]) -> bool:
    return all(map(lambda i: i > 0, data))


def to_ln(data: list[float]) -> bool:
    return list(map(lambda i: math.log(i), data))


def reliability_of_approximation(y_points: list[float], phi_points: list[float]) -> float:
    phi_mean = sum(phi_points) / len(phi_points)
    top = sum([(y - phi)**2 for y, phi in zip(y_points, phi_points)])
    bottom = sum([(y - phi_mean)**2 for y in y_points])

    return 1 - top / bottom

# data = [ (rel, name), ...] 
# where `rel` is reliability (float) and `name` is approxiamtion methods' name (str)
def max_reliability(data: list) -> tuple:
    max_rel = -1
    result = ()
    for pair in data:
        if pair[0] > max_rel:
            max_rel = pair[0]
            result = pair

    return result

# data = [ (rel, name), ...]
def print_reliabilities(data: list) -> None:
    for pair in data:
        print(f"{pair[1].capitalize()}: {pair[0]}")