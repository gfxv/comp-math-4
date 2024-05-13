import math

def all_exp_safe(data: list[float]) -> bool:
    return all(map(lambda i: i > 0, data))


def to_ln(data: list[float]) -> bool:
    return list(map(lambda i: math.log(i), data))