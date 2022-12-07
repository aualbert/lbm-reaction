import imageio
import numpy as np

# import borders,        species and cells from an image
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
