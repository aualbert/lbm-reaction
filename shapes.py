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

#maps colours (r + 256 * g + 256 * 256 * b) / 100
def colour_mapper(colour):
    return float(colour[0]) / 100 + float(colour[1]) * 256 / 100 + float(colour[2]) * 256 * 256 / 100

#black is border
def import_border(imagename, x, y):
    im = imageio.imread(imagename)
    im = im[:y,:x,:]
    answ = np.full(im.shape[:2], False)
    black = colour_mapper([0,0,0])
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if colour_mapper(im[i][j]) == black:
                answ[i][j] = True
    return answ

#any colour but black and white is nutrients. The witer the color is(the closer to 255/255/255) the more nutriens there are
def set_nutrients(imagename, x, y):
    im = imageio.imread(imagename)
    im=im[:y,:x,:]
    answ = np.zeros((im.shape[0], im.shape[1], 9))
    border = colour_mapper([255,255,255])
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            #if color is black than ignore it as it is border
            colour_sum = colour_mapper(im[i][j])
            if colour_sum == border:
                continue
            answ[i][j][8] = colour_sum
    return answ