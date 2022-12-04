import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import shapes

"""
Lattice Boltzmann method with arbitrary reactions to model E. coli growth.
Based on code from Philip Mocz (2020) Princeton Univeristy, @PMocz.
"""


def main():

    """
    Simulation parameters for the 2DQ9 model.
    See https://www.sciencedirect.com/science/article/pii/S0898122111004731
    for more information on the relations between these parameters.
    """
    
    """
    To observe turbulences, the size of the channel should be 3x1.
    """
    Nx = 450  # 10^-4 m, length of the channel
    Ny = 150  # 10^-4 m, width of the channel
    dx = 1  # 10^-4 m, lattice spacing
    L = 40  # ~ 10^-4 m, caracteristic size of obstacles
    Nt = 500  # number of steps
    dt = 1  # 10^-3 s, simulation timestep

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
    flow_nut = 0.5  # quantity of nutrient particles flowing in at each step 

    """
    Initial conditions.
    """
    rho0 = 100  # average density for initialisation
    F = np.ones((Ny, Nx, NL))  # fluid
    G = np.zeros((Ny, Nx, NL))  # nutrients
    C = np.zeros((Ny,Nx,NL))

    # Initialisation of fluid
    np.random.seed(40)
    F += 0.1 * np.random.randn(Ny, Nx, NL)
    X, Y = np.meshgrid(range(Nx), range(Ny))
    F[:, :, 3] += 2 * (1 + 0.2 * rd.random())
    rho = np.sum(F, 2)
    for i in idxs:
        F[:, :, i] *= rho0 / rho

    # Initialisation of cells
    G = np.zeros((Ny,Nx,NL))

    for x in range((Nx//2 + Ny//4), (Nx//2 + Ny//4 + Nx//10)):
        for y in range ((Ny//2 - Ny//10),(Ny//2 + Ny//10)):
            C[y,x,:] = 1
    for x in range((Ny//4), (Ny//4 + Nx//10)):
        for y in range ((Ny//3 - Ny//10),(Ny//3 + Ny//10)):
            C[y,x,:] = 1
    for x in range(5, Nx//10):
        for y in range ((2 * Ny//3 - Ny//10),(2* Ny//3 + Ny//10)):
            C[y,x,:] = 1

    # Initialisation of nutrients and obstacles
    obstacles, G = shapes.import_image("input_images/image1.png", Nx, Ny)
    X, Y = np.meshgrid(range(Nx), range(Ny))
    obstacles = (
        (Y < 1)
        | (Ny - Y < 1)
        | obstacles
        # | shapes.circle(X, Y, Nx / 4, Ny / 2, Ny / 8)
        # | shapes.square(X, Y, Nx / 2, Ny / 2, Ny / 2)
    )

    """
    Animation parameters
    """
    labels = ["Vorticity", "Density", "Horizontal Speed", "Nutrients Concentration"]
    fig, axs = plt.subplots(4)
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
        #simulate the flow -> extend the array
        C = np.pad(C, ((0,0),(1,1),(0,0)),'edge')
        C = np.pad(C,((0,0),(0,1),(0,0)))

        # Drift
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            C[:, :, i] = np.roll(C[:, :, i], cx, axis=1)
            C[:, :, i] = np.roll(C[:, :, i], cy, axis=0)

        # cut the array
        C = C[:, 1:Nx+1, :]

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
        rho = np.sum(F, 2)
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
        Clc = np.sum(C,2)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Ceq[:, :, i] = (
                Clc
                * w
                * (
                    1
                    + icsc * (cx * ux + cy * uy)
            ))

        C += -(dt / tauc) * (C - Ceq)


        # Apply boundary
        F[obstacles, :] = bndryF
        G[obstacles, :] = bndryG
        C[obstacles, :] = bndryC

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
            im3 = axs[3].imshow(nutri, cmap="hot_r", vmin=0, vmax=3000)

            #Cells
            cells = np.ma.array(np.sum(C,2), mask = obstacles)
            im4 = axs[4].imshow(cells, cmap="ocean_r",vmin = 0, vmax = 15)

            ims.append([im0, im1, im2, im3, im4])

    """
    Saving animation
    """
    fig.colorbar(im0, ax=axs[0], location='left')
    fig.colorbar(im1, ax=axs[1], location='left')
    fig.colorbar(im2, ax=axs[2], location='left')
    fig.colorbar(im3, ax=axs[3], location='left')
    fig.colorbar(im4, ax=axs[4], location='left')

    print("\ncreating animation")
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    print("saving animation")
    ani.save("output.gif")

    return 0

if __name__ == "__main__":
    main()
