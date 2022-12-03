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
    return (X - x) ** 2 / a**2 + (Y - y) ** 2 / b**2 < r**2


# map rgb color to greyscale
def rgb_to_grey(rgb):
    return 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]


# import borders and nutrients from an image
def import_image(file, Nx, Ny):
    im = imageio.imread(file)[:Ny, :Nx, :] 
    im = (1000 * (255 - rgb_to_grey(im)) / 255).round() 
    map_nutrients = np.zeros((Ny, Nx, 9))
    map_nutrients[:, :, 8] = im
    map_borders = np.matrix(im == 1000)
    return map_borders, map_nutrients
