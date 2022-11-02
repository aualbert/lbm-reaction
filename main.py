import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import numpy as np
import shapes

"""
Lattice Boltzmann method with arbitrary reactions to model E. coli growth.
Based on code from Philip Mocz (2020) Princeton Univeristy, @PMocz.
"""


def main():

    # Simulation parameters
    Nx = 400  # length
    Ny = 100  # width
    rho0 = 100  # average density
    tau = 0.6  # relaxation factor
    Nt = 2000  # number of timesteps
    icsc = 3 # see paper on biofilms 1/cs^2 -> influes on viscosity
    flow = 0.05

    # Lattice speeds / weights for D2Q9
    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array(
        [4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36] )  # sums to 1

    # Initial Conditions
    F = np.ones((Ny, Nx, NL))  # * rho0 / NL
    np.random.seed(42)
    F += 0.01 * np.random.randn(Ny, Nx, NL)
    X, Y = np.meshgrid(range(Nx), range(Ny))
    F[:, :, 3] += 2 * (1 + 0.2 * np.cos(2 * np.pi * X / Nx * 4))
    rho = np.sum(F, 2)
    for i in idxs:
        F[:, :, i] *= rho0 / rho

    # Initial condition Nutrients
    G = np.zeros((Ny,Nx,NL))

    # Obstacles
    X, Y = np.meshgrid(range(Nx), range(Ny))
    obstacles = (
        (Y < 1)
        | (Ny - Y < 1)
        | shapes.circle(X, Y, Nx / 4, Ny / 2, Ny / 8)
        | shapes.square(X, Y, Nx / 2, Ny / 2, Ny / 2)
    )

    # Animation parameters
    fig, axs = plt.subplots(4)
    plt.tight_layout()
    for i in range(4):
        axs[i].imshow(~obstacles, cmap="gray", alpha=0.3)
    ims = []

    # Simulation Main Loop
    for it in range(Nt):
        print("\r", it, "/", Nt, end="")

        # Drift
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

         #simulate the flow -> extend the array
        G = np.pad(G, ((0,0),(1,1),(0,0)),'edge')
        G = np.pad(G,((0,0),(0,1),(0,0)))
        G[:10,int(Nx/2),3] += flow

        # Drift Nutrients
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            G[:, :, i] = np.roll(G[:, :, i], cx, axis=1)
            G[:, :, i] = np.roll(G[:, :, i], cy, axis=0)

        # cut the array
        G = G[:, 1:Nx+1, :]

        # Set reflective boundaries
        bndryF = F[obstacles, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
        bndryG = G[obstacles, :]
        bndryG = bndryG[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

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
                    + icsc * (cx * ux + cy * uy)
                    + icsc**2 * (cx * ux + cy * uy) ** 2 / 2
                    - icsc * (ux**2 + uy**2) / 2
            ))

        F += -(1.0 / tau) * (F - Feq)

        # Apply collisions nutrients
        Geq = np.zeros(G.shape)
        Cl = np.sum(G,2)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Geq[:, :, i] = (
                Cl
                * w
                * (
                    1
                    + icsc * (cx * ux + cy * uy)
            ))

        G += -(1.0 / tau) * (G - Geq)

        # Apply boundary
        F[obstacles, :] = bndryF
        G[obstacles, :] = bndryG

        # Plot every 10 steps
        if (it % 10) == 0:

            # vorticity
            vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (
                np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)
            )
            vorticity = np.ma.array(vorticity, mask=obstacles)
            im0 = axs[0].imshow(vorticity, cmap="bwr")

            # density
            density = np.ma.array(rho, mask=obstacles)
            im1 = axs[1].imshow(density, cmap="Blues")

            # horizontal speed
            speed = np.ma.array(ux, mask=obstacles)
            im2 = axs[2].imshow(speed, cmap="bwr")

            #Nutrients
            nutri = np.ma.array(np.sum(G,2), mask = obstacles)
            im3 = axs[3].imshow(nutri, cmap="hot_r")
            ims.append([im0, im1, im2, im3])


    # Save figure
    print("\ncreating animation")
    ani = animation.ArtistAnimation(
        fig, ims, interval=50, blit=False, repeat_delay=1000
    )
    print("saving animation")
    ani.save("output.gif")

    return 0


if __name__ == "__main__":
    main()
