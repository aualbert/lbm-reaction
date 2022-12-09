from itertools import chain
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import random as rd
from numba import jit

"""
Lattice Boltzmann method with arbitrary reactions to model E. coli growth.
Based on code from Philip Mocz (2020) Princeton Univeristy, @PMocz.
"""


def CustomCmap(to_rgb):
    """
    Convert a rgb color in the range 0-255 in a linear colormap
    """
    r, g, b = to_rgb
    r = r / 255
    g = g / 255
    b = b / 255
    cdict = {
        "red": ((0, 1, 1), (1, r, r)),
        "green": ((0, 1, 1), (1, g, g)),
        "blue": ((0, 1, 1), (1, b, b)),
    }
    return colors.LinearSegmentedColormap("custom_cmap", cdict)


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        # Note also that we must extrapolate beyond vmin/vmax
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.0]
        return np.ma.masked_array(np.interp(value, x, y, left=-np.inf, right=np.inf))

    def inverse(self, value):
        y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)

def run_simulation(
    Ny: int,
    Nx: int,
    Nt: int,
    obstacles,
    species,
    species_desc,
    cells,
    cells_desc,
    save_path,
    ):
    """
    Simulation of the Lattice Boltzmann 2DQ9 model
    with arbitrary reactions between species and cells.

    To observe turbulences, the size of the channel should be 3x1.
    """
    Nx = Nx  # 10^-4 m, length of the channel
    Ny = Ny  # 10^-4 m, width of the channel
    L = 40  # ~ 10^-4 m, caracteristic size of obstacles
    dx = 1  # 10^-4 m, lattice spacing
    Nt = Nt  # number of steps
    dt = 1  # 10^-3 s, simulation timestep

    NL = 9  # number of directions
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array(
        [4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36]
    )  # sums to 1

    """
    Coefficients for cells and nutrients behaviours
    """
    alpha = 0.000001 #about 10^-6
    beta = 0.0001  # coefficient for reflection of water on cells (about 10^-4)
    gammaReproduction = 0.001  # fractions of cells to reproduce
    gammaDeath = 0.0001  # fractions of cells to die

    """
    The relaxation factor for fluid sould be contained between 0.6 and 1.
    0.9080 is a good choice for accuracy and stability.
    """
    tau = 0.9080  # 10^-1 s, relaxation factor for fluid and disolved nutrients
    nu = tau - 1 / 2  # (* c^2) 10^-9 m^2/s, kinematic viscosity
    tau_cell = 0.6  # 10^-1 s, relaxation factor for cells

    """
    To observe laminar flow, the Reynolds number
    should be contained between 100 and 150.
    It is not clear yet how this relates to the other parameters
    """
    Re = 100  # dimensionless , Reynolds number, Re = VL/nu
    V = (Re * nu) / L  # 10^-5 m/s caracteristic speed of the fluid

    """
    For the simulation not to break, flow << 1 is required.
    """
    flow = 0.01  # quantity of fluid particules flowing in at each step
    flow_nut = 0.01  # quantity of nutrient particles flowing in at each step

    """
    Initial conditions.
    """
    # masse volumique de l'eau: ~10^-26kg/unité
    # donc donc une unité = 10^24 molécules
    rho0 = 1000  # 10^12 kg/m³ average density for initialisation
    F = np.ones((Ny, Nx, NL))  # fluid
    G = species[0]  # nutrients
    C = cells[0]  # cells

    # Initialisation of fluid
    np.random.seed(40)
    F += 0.1 * np.random.randn(Ny, Nx, NL)
    F[:, :, 3] += 2 * (1 + 0.2 * rd.random())
    rho = np.sum(F, 2)
    for i in idxs:
        F[:, :, i] *= rho0 / rho

    def drift_particles(F) :
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)
        return F

    def simulate_flow_fluid (F) :
        """
        Simulate the flow with pseudo physical boundaries ""
        """
        F = np.pad(F, ((0, 0), (1, 1), (0, 0)), "edge") #extend the array
        F[:, 0, :] = F[:, 3, :]
        F[:, -1 , :] = F[:, -4 , :]
        F[:, 0, 3] += flow
        F = drift_particles(F) # drift
        F = F[:, 1 : Nx + 1, :] # cut the array
        return F

    def simulate_flow_elements(G, flow_element) :
        G = np.pad(G, ((0, 0), (1, 1), (0, 0)), "edge")
        G[:, 0, 3] += flow_element
        G = drift_particles(G)
        G = G[:, 1 : Nx + 1, :]
        return G

    @jit(nopython = True)
    def apply_collisions_water(F, rho, ux, uy) :
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:, :, i] = (
                rho
                * w
                * (
                    1
                    + 3 * (cx * ux + cy * uy)
                    + (9 * (cx * ux + cy * uy) ** 2) / 2
                    - 3 * (ux**2 + uy**2) / 2
                )
            )
        return (F - ((dt / tau) * (F - Feq)))

    @jit(nopython = True)
    def calculate_fluids_variables(F) :
        rho = np.multiply(np.sum(F, 2), (1 + alpha * np.sum(C, 2)))
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho
        return (rho, ux, uy)

    @jit(nopython = True)
    def apply_collisions_elements(G,rho,ux,uy) :
        Geq = np.zeros(G.shape)
        Cl = np.sum(G, 2)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Geq[:, :, i] = Cl * w  * (1 + 3 * (cx * ux + cy * uy))
        return G -(dt / tau) * (G - Geq)

    def get_bndry (F):
        bndryF = F[obstacles, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
        return bndryF


    def bounce_on_cells(F, C):
        cellsConcentration = np.sum(C, 2)
        cc = np.empty((Ny, Nx, NL))
        for i in range(NL):
            cc[:, :, i] = cellsConcentration
        F = np.multiply((1 - beta * cc), F) + beta * np.multiply(
            cc, F[:, :, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
        )
        return F

    def simulate_cells_life (G,C):
        # Simulate the feeding and reproductions of bacterias
        nbOfNewBacteria = np.minimum(C, G) * gammaReproduction
        C += nbOfNewBacteria
        G -= nbOfNewBacteria
        # Simulate the death of cells
        nbOfDeadBacteria = C * gammaDeath
        C -= nbOfDeadBacteria
        return (G, C)

    # Initialisation of matplotlib
    labels = [
        "vorticity",
        "density (kg/m$^3$)",
        "horiz. speed",
    ]

    species_cmaps = list(
        map(
            lambda element: CustomCmap(element[1]),
            species_desc,
        )
    )

    cells_cmaps = list(
        map(
            lambda element: CustomCmap(element[1]),
            cells_desc,
        )
    )

    norm_density = MidpointNormalize(vmin=800, vcenter=1000, vmax=1200)

    size = max(3, len(cells_desc), len(species_desc))
    fig, axs = plt.subplots(3, size)
    plt.rcParams.update({"axes.titlesize": "medium"})
    ims = []

    for i in range(3):
        for j in range(size):
            axs[i, j].axis("off")
            axs[i, j].axis("off")

    ite = zip(range(3), 3 * [0])
    ite = chain(ite, zip(range(len(species_desc)), size * [1]))
    ite = chain(ite, zip(range(len(cells_desc)), size * [2]))

    for i, j in ite:
        axs[i, j].axis("on")
        axs[i, j].axes.get_yaxis().set_visible(False)
        axs[i, j].axes.set_xlabel("10^-4 m")
        axs[i, j].imshow(obstacles, cmap="gray", alpha=0.3)
        if i == 0:
            axs[i, j].set_title(labels[j])
        elif i == 1:
            axs[i, j].set_title("specie " + species_desc[j][0] + " concn.")
        else:
            axs[i, j].set_title("cell " + cells_desc[j][0] + " concn.")

    """
    Simulation main loop
    """
    for it in range(Nt):
        print("\r", it, "/", Nt, end="")

        # Flows
        F = simulate_flow_fluid(F)
        G = simulate_flow_elements(G, flow_nut)
        C = simulate_flow_elements(C,0)

        # Solid Boundaries
        bndryF = get_bndry(F)
        bndryG = get_bndry(G)
        bndryC = get_bndry(C)

        # Collisions
        (rho, ux, uy) = calculate_fluids_variables(F)
        F = apply_collisions_water(F, rho, ux, uy)
        G = apply_collisions_elements(G, rho, ux, uy)
        C = apply_collisions_elements(C, rho, ux, uy)

        # Apply boundary
        F[obstacles, :] = bndryF
        G[obstacles, :] = bndryG
        C[obstacles, :] = bndryC

        # Specific cells actions
        F = bounce_on_cells(F, C)
        (G, C) = simulate_cells_life(G, C)

        """
        Simulation plotting every 10 steps
        """
        if (it % 10) == 0:

            # vorticity
            vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (
                np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)
            )
            vorticity = np.ma.array(vorticity, mask=obstacles)
            im0 = axs[0, 0].imshow(
                vorticity, cmap="bwr", label="Vorticity", vmin=-0.3, vmax=0.3
            )

            # density
            density = np.ma.array(rho, mask=obstacles)
            im1 = axs[0, 1].imshow(density, cmap="Blues", norm=norm_density)

            # horizontal speed
            speed = np.ma.array(ux, mask=obstacles)
            im2 = axs[0, 2].imshow(speed, cmap="bwr", vmin=-0.5, vmax=0.5)

            # Nutrients
            nutri = np.ma.array(np.sum(G, 2), mask=obstacles)
            im3 = axs[1, 0].imshow(nutri, cmap=species_cmaps[0], vmin=0, vmax=30)

            # Cells
            cells = np.ma.array(np.sum(C, 2), mask=obstacles)
            im4 = axs[2, 0].imshow(cells, cmap=cells_cmaps[0], vmin=0, vmax=15)

            ims.append([im0, im1, im2, im3, im4])

    """
    Saving animation
    """
    fig.colorbar(im0, ax=axs[0, 0], location="left")
    fig.colorbar(im1, ax=axs[0, 1], location="left")
    fig.colorbar(im2, ax=axs[0, 2], location="left")
    for i in range(len(species_desc)):
        fig.colorbar(im3, ax=axs[1, i], location="left")
    for i in range(len(cells_desc)):
        fig.colorbar(im4, ax=axs[2, i], location="left")

    print("\ncreating animation")
    # Same speed as the simulation
    ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat_delay=1000)

    print("saving animation")
    ani.save(save_path)

    return 0
