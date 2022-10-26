def circle(X, Y, x, y, r):
    return (X - x) ** 2 + (Y - y) ** 2 < r**2


def square(X, Y, x, y, l):
    return ((X - x) ** 2 < (l / 2) ** 2) & ((Y - y) ** 2 < (l / 2) ** 2)
