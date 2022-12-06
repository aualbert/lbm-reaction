import imageio
import numpy as np

# import borders,        species and cells from an image
def import_image(image, species, cells, Nx, Ny):
    im = imageio.imread(image)

    # get dimensions of image if unprovided
    if Nx == 0 or Ny == 0:
        (Nx, Ny, _) = im.shape
    im = im[:Nx, :Ny, 0:3]

    # get obstacles
    obstacles = np.squeeze(np.take(np.array(im == [0, 0, 0]), indices=[0], axis=2))
    obstacles[0, :] = True
    obstacles[-1, :] = True

    def get_color(element):
        color = element[1]
        lattice = np.zeros((Nx, Ny, 9))
        lattice[:, :, 8] = np.squeeze(
            np.take(np.array(im == color), indices=[0], axis=2)
        )
        return lattice

    # get species
    species = list(map(lambda a: 25 * a, map(get_color, species)))

    # get cells
    cells = list(map(lambda a: 10 * a, map(get_color, cells)))

    return (Nx, Ny, obstacles, species, cells)
