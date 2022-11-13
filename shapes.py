import random
import imageio
import numpy as np
def circle(X, Y, x, y, r):
    return (X - x) ** 2 + (Y - y) ** 2 < r**2

def square(X, Y, x, y, l):
    return ((X - x) ** 2 < (l / 2) ** 2) & ((Y - y) ** 2 < (l / 2) ** 2)

def rectangle(X, Y, x, y, xl, yl):
    return ((X - x) ** 2 < (xl / 2) ** 2) & ((Y - y) ** 2 < (yl / 2) ** 2)

def ellipse(X, Y, x, y, a, b, r):
    return (X - x) ** 2 / a ** 2 + (Y - y) ** 2 / b ** 2 < r ** 2

def test(imagename):
    im=imageio.imread(imagename)
    answ = np.full(X.shape, False)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if not np.all(im[i][j]):
                answ[i][j] = True
    return answ