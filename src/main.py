import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import random as rd

"""
Lattice Boltzmann method with arbitrary reactions to model E. coli growth.
Based on code from Philip Mocz (2020) Princeton Univeristy, @PMocz.
"""


def run_simulation(Ny: int, Nx: int, Nt: int, obstacles, species, cells, save_path):
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
    alpha = 0.000001  # coefficient of cell volume over buckets volume (about 10 -6)
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
    It is not yet clear how it relates to the initial density
    """
    Re = 100  # dimensionless, Reynolds number, Re = VL/nu
    V = (Re * nu) / L  # 10^-5 m/s caracteristic speed of the fluid

    """
    For the simulation not to break, flow << 1 is required.
    """
    flow = 0.001  # quantity of fluid particules flowing in at each step
    flow_nut = 0.01  # quantity of nutrient particles flowing in at each step

    """
    Initial conditions.
    """
    rho0 = 100  # average density for initialisation
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

    labels = [
        "Vorticity",
        "Density",
        "Horizontal Speed",
        "Nutrients Concentration",
        "Cells concentration",
    ]
    fig, axs = plt.subplots(5)
    plt.tight_layout()
    for i in range(5):
        axs[i].imshow(~obstacles, cmap="gray", alpha=0.3)
        axs[i].set_title(labels[i])
        axs[i].axes.get_xaxis().set_visible(False)
        axs[i].axes.get_yaxis().set_visible(False)
    ims = []

    """
    Simulation main loop
    """
    for it in range(Nt):
        print("\r", it, "/", Nt, end="")

        """
        Drifting fluid
        """
        # simulate the flow -> extend the array
        F = np.pad(F, ((0, 0), (1, 1), (0, 0)), "edge")
        F = np.pad(F, ((0, 0), (0, 1), (0, 0)))
        F[:, 0, :] = F[:, 3, :]
        F[:, Nx + 1, :] = F[:, Nx - 2, :]
        F[:, 0, 3] += flow

        # Drift fluid particles
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

        # cut the array
        F = F[:, 1 : Nx + 1, :]

        """
        Drifting nutrients
        """
        # simulate the flow -> extend the array
        G = np.pad(G, ((0, 0), (1, 1), (0, 0)), "edge")
        G = np.pad(G, ((0, 0), (0, 1), (0, 0)))
        G[:, 0, 3] += flow_nut

        # Drift nutrients
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            G[:, :, i] = np.roll(G[:, :, i], cx, axis=1)
            G[:, :, i] = np.roll(G[:, :, i], cy, axis=0)

        # cut the array
        G = G[:, 1 : Nx + 1, :]

        """
        Drifting cells
        """
        # simulate the flow -> extend the array
        C = np.pad(C, ((0, 0), (1, 1), (0, 0)), "edge")
        C = np.pad(C, ((0, 0), (0, 1), (0, 0)))

        # Drift
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            C[:, :, i] = np.roll(C[:, :, i], cx, axis=1)
            C[:, :, i] = np.roll(C[:, :, i], cy, axis=0)

        # cut the array
        C = C[:, 1 : Nx + 1, :]

        """
        Colisions
        """
        # Set reflective boundaries
        bndryF = F[obstacles, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
        bndryG = G[obstacles, :]
        bndryG = bndryG[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
        bndryC = C[obstacles, :]
        bndryC = bndryC[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

        # Calculate fluid variables
        rho = np.multiply(np.sum(F, 2), (1 + alpha * np.sum(C, 2)))
        ##rho = np.sum(F, 2)
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho

        # Apply Collision
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

        F += -(dt / tau) * (F - Feq)

        # Apply collisions nutrients
        Geq = np.zeros(G.shape)
        Cl = np.sum(G, 2)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Geq[:, :, i] = Cl * w * (1 + 3 * (cx * ux + cy * uy))

        G += -(dt / tau) * (G - Geq)

        # Apply collisions cells
        Ceq = np.zeros(C.shape)
        Clc = np.sum(C, 2)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Ceq[:, :, i] = Clc * w * (1 + 3 * (cx * ux + cy * uy))

        C += -(dt / tau_cell) * (C - Ceq)

        # Apply boundary
        F[obstacles, :] = bndryF
        G[obstacles, :] = bndryG
        C[obstacles, :] = bndryC

        # bounce of water on cells
        cellsConcentration = np.sum(C, 2)
        cc = np.empty((Ny, Nx, NL))
        for i in range(NL):
            cc[:, :, i] = cellsConcentration
        F = np.multiply((1 - beta * cc), F) + beta * np.multiply(
            cc, F[:, :, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
        )

        # Simulate the feeding and reproductions of bacterias
        nbOfNewBacteria = np.minimum(C, G) * gammaReproduction
        # print(nbOfNewBacteria)
        C += nbOfNewBacteria
        G -= nbOfNewBacteria

        # Simulate the death of cells
        nbOfDeadBacteria = C * gammaDeath
        C -= nbOfDeadBacteria

        """
        Simulation plotting every 10 steps
        """
        if (it % 10) == 0:

            # vorticity
            vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (
                np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)
            )
            vorticity = np.ma.array(vorticity, mask=obstacles)
            im0 = axs[0].imshow(
                vorticity, cmap="bwr", label="Vorticity", vmin=-0.3, vmax=0.3
            )

            # density
            density = np.ma.array(rho, mask=obstacles)
            im1 = axs[1].imshow(density, cmap="Blues", vmin=0, vmax=140)

            # horizontal speed
            speed = np.ma.array(ux, mask=obstacles)
            im2 = axs[2].imshow(speed, cmap="bwr", vmin=-0.5, vmax=0.5)

            # Nutrients
            nutri = np.ma.array(np.sum(G, 2), mask=obstacles)
            im3 = axs[3].imshow(nutri, cmap="hot_r", vmin=0, vmax=30)

            # Cells
            cells = np.ma.array(np.sum(C, 2), mask=obstacles)
            im4 = axs[4].imshow(cells, cmap="ocean_r", vmin=0, vmax=15)

            ims.append([im0, im1, im2, im3, im4])

    """
    Saving animation
    """
    fig.colorbar(im0, ax=axs[0], location="left")
    fig.colorbar(im1, ax=axs[1], location="left")
    fig.colorbar(im2, ax=axs[2], location="left")
    fig.colorbar(im3, ax=axs[3], location="left")
    fig.colorbar(im4, ax=axs[4], location="left")

    print("\ncreating animation")
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    print("saving animation")
    ani.save(save_path)

    return 0
