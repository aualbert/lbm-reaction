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


#black is border
def import_border(imagename, x, y):
    im = imageio.imread(imagename)
    im = im[:y,:x,:]
    answ = np.full(im.shape[:2], False)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if not np.all(im[i][j]):
                answ[i][j] = True
    return answ

#maps colours to nutrients but i am not sure if this mapping is good
def colour_mapper(colour):
    return int(colour[0]) + int(colour[1]) + int(colour[2])

#any colour but black and white is nutrients. The witer the color is(the closer to 255/255/255) the more nutriens there are
def set_nutrients(imagename, x, y):
    im = imageio.imread(imagename)
    im=im[:y,:x,:]
    answ = np.zeros((im.shape[0], im.shape[1], 9))
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            #if color is black than ignore it as it is border
            colour_sum = colour_mapper(im[i][j])
            if colour_sum == 765:
                continue
            answ[i][j][8] = colour_sum
    return answ