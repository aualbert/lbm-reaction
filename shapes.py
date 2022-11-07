import random

def circle(X, Y, x, y, r):
    return (X - x) ** 2 + (Y - y) ** 2 < r**2

def square(X, Y, x, y, l):
    return ((X - x) ** 2 < (l / 2) ** 2) & ((Y - y) ** 2 < (l / 2) ** 2)

def rectangle(X, Y, x, y, xl, yl):
    return ((X - x) ** 2 < (xl / 2) ** 2) & ((Y - y) ** 2 < (yl / 2) ** 2)

def ellipse(X, Y, x, y, a, b, r):
    return (X - x) ** 2 / a ** 2 + (Y - y) ** 2 / b ** 2 < r**2


def randfig(X, Y, x, y, Nx, Ny):
    #state = random.randint(0, 3)
    state = 3
    if state == 0:
        return circle(X, Y, x, y, r)
    elif state == 1:
        div = random.randint(2,6)
        return square(X,Y,x,y, min(Nx, Ny) / div)
    elif state == 2:
        divx = random.randint(2, 6)
        divy = random.randint(2,6)
        return rectangle(X,Y,x,y,Nx / divx, Ny / divy)
    elif state == 3:
        a = random.randint(2,4)
        b = random.randint(1,4)
        return ellipse(X,Y,x,y,a,b , 3 * (a ** 2 + b ** 2))