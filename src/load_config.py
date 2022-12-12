import imageio
import numpy as np
from numba import jit

# import borders, species and cells from an image
def import_image(image, species, cells, Nx, Ny):
    im = imageio.imread(image)

    # get dimensions of image if unprovided
    if Nx == 0 or Ny == 0:
        (Nx, Ny, _) = im.shape
    im = im[:Nx, :Ny, 0:3]

    def get_color(element):
        color = element[1]
        lattice = np.zeros((Nx, Ny, 9))
        lattice[..., 8] = (
            (im[..., 0] == color[0])
            & (im[..., 1] == color[1])
            & (im[..., 2] == color[2])
        )
        return lattice

    # get obstacles
    obstacles = (im[..., 0] == 0) & (im[..., 1] == 0) & (im[..., 2] == 0)
    obstacles[0, :] = True
    obstacles[-1, :] = True

    # get species
    species = np.array(list(map(lambda a: 25 * a, map(get_color, species))))

    # get cells
    cells = np.array(list(map(lambda a: 10 * a, map(get_color, cells))))

    return (Nx, Ny, obstacles, species, cells)


def import_reactions(former_reactions, species, cells) :
    n = len (former_reactions)
    alpha = 0.0001
    get_element = {}     # for a key k which is the name of an element contains whether it is a cell or a species and its position in tha array
    nb_species = len(species)
    nb_cells = len(cells)
    for i in range (nb_species) :
        get_element[(species[i][0])] = (True, i) # true for species
    for i in range (nb_cells) :
        get_element[(cells[i][0])] =  (False, i) # false for cells
    def compute_reactions(G,C) :
        Nx, Ny, Nl = np.shape(G[0])
        for i in range (n): #for each reaction
            # determine the speed of the reactions
            speed = np.full((Nx, Ny, Nl) , alpha * former_reactions[i]["rate"][0]) # speed coefficient
            for reactant, coefSpeed in zip(former_reactions[i]["reactants"], former_reactions[i]["rate"][1]) :
                (s, j) = get_element.get(reactant[1])
                if s :
                    speed *= np.power (G[j], coefSpeed)
                else :
                    speed *= np.power (C[j], coefSpeed)
            #delete reactants
            for reactant in former_reactions[i]["reactants"]:
                (s,j) = get_element.get(reactant[1])
                if s :
                    G[j] -= speed * reactant[0]
                else :
                    C[j] -= speed * reactant[0]
            #add products
            for product in former_reactions[i]["products"] :
                (s,j) = get_element.get(product[1])
                if s :
                    G[j] += speed * product[0]
                else :
                    C[j] += speed * product[0]
        return (G,C)
    return compute_reactions
