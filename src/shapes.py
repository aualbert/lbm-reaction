# predefined shapes used for testing 

def circle(X, Y, x, y, r):
    return (X - x) ** 2 + (Y - y) ** 2 < r**2


def square(X, Y, x, y, l):
    return ((X - x) ** 2 < (l / 2) ** 2) & ((Y - y) ** 2 < (l / 2) ** 2)


def rectangle(X, Y, x, y, xl, yl):
    return ((X - x) ** 2 < (xl / 2) ** 2) & ((Y - y) ** 2 < (yl / 2) ** 2)


def ellipse(X, Y, x, y, a, b, r):
    return (X - x) ** 2 / a**2 + (Y - y) ** 2 / b**2 < r**2

