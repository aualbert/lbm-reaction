import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import numpy as np
import shapes
import random as rd
import math

"""
Lattice Boltzmann method with arbitrary reactions to model E. coli growth.
Based on code from Philip Mocz (2020) Princeton Univeristy, @PMocz.
"""


"""
0 -----> x
|
|
|
V
y
"""


def main():

    # Simulation parameters
    Nx = 400 # 400  # length
    Ny = 150 # 150  # width
    rho0 = 100  # average density
    tau = 0.65  # 0.6 relaxation factor please keep it between 0.6 and 1
    taul = 0.6 # relaxation factor nutrient
    Nt = 600 # 500  # number of timesteps
    icsc = 3 # see paper on biofilms 1/cs^2 -> influes on viscosity
    Lflow= 0.001
    Nflow = 1
    dt = 1
    sqs = 1 # square size
    g = 9.81 # gravity constante
    h = 0.001 # high of the water
    minmvt = 0.1 # if the force is more than minmvt then the cell is pushed
    minmvt_adh = 0.01 # if there is adhesion (close to obstacles), the minmvt value is higher
    D = 1 # diameter of ecolis
    Kf = -3*np.pi*0.001*D  # Constante for fluid friction (Fot water at 20 °C for 1 to 100 bar, we have μ = 1 × 10−3 Pa s)
    mc = 3 # maximal number of cells per square
    adhreach = 2 # obstacles create adhesion to cells distant to them of less than adhreach
    maxd = 300 # when too many cells in a square, the calculus of density fails, so we put a big value maxd instead

    vol = sqs**2 * h # volume of a square
    vc = 4/3 * np.pi * (D/2)**3 # volume of ecolis

    # Lattice speeds / weights for D2Q9
    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array(
        [4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36]
    )  # sums to 1

    # Initial Conditions
    F = np.ones((Ny, Nx, NL))  # * rho0 / NL
    np.random.seed(40)
    F += 0.1 * np.random.randn(Ny, Nx, NL)
    X, Y = np.meshgrid(range(Nx), range(Ny))
    F[:, :, 3] += 2 * (1 + 0.2 * rd.random())
    rho = np.sum(F, 2)
    for i in idxs:
        F[:, :, i] *= rho0 / rho

    # Initial condition of nutrients and obstacles
    G = np.zeros((Ny, Nx, NL))
    obstacles, G = shapes.import_image("input_images/image1.png", Nx, Ny)
    X, Y = np.meshgrid(range(Nx), range(Ny))
    obstacles = (
        (Y < 1)
        | (Ny - Y < 2)
        | obstacles
        # | shapes.circle(X, Y, Nx / 4, Ny / 2, Ny / 8)
        # | shapes.square(X, Y, Nx / 2, Ny / 2, Ny / 2)
    )
    obstacles2 = np.ones((Ny+2, Nx+2), dtype=bool) # extended version of obstacles (+1 on each side)
    obstacles2[1:Ny+1, 1:Nx+1] = obstacles
    for y in range(2,Ny):
        obstacles2[y,0] = False
        obstacles2[y,Nx+1] = False

    close_obs = obstacles2 # squares closes to an obstacle
    for x in range(Nx+2):
        for i in range(adhreach):
            close_obs[1+i,x] = True
            close_obs[Ny-i,x] = True
    for x in range(1+adhreach,Nx-adhreach-1):
        for y in range(1+adhreach,Ny-adhreach-1):
            if obstacles2[y,x]:
                for i in range(-adhreach,adhreach):
                    for j in range(-adhreach,adhreach):
                        if y+i > 0 and y+i < Ny+1 and x+j > 0 and x+j < Nx+1:
                            close_obs[y+i,x+j] = True

    # Forces
    Forces = np.zeros((Ny,Nx,2)) # vectors of the force
    hsq2 = (math.sqrt(2))/2
    unary_vect = np.array([[0,1], [hsq2,hsq2], [1,0], [hsq2, -hsq2], [0,-1], [-hsq2,-hsq2], [-1,0], [-hsq2,hsq2]])
    #unary_vect = np.array([[1,0], [hsq2,hsq2], [0,1], [-hsq2, hsq2], [-1,0], [-hsq2,-hsq2], [0,-1], [hsq2,-hsq2]])
    sq2 = math.sqrt(2)
    dx = sqs * np.array([1,sq2,1,sq2,1,sq2,1,sq2])

    # Cells
    C = np.zeros((Ny,Nx))
    for x in range(Nx):
        for y in range(Ny):
            if (rd.random() > 0.8):
                C[y,x] = 1
    C2 = np.zeros((Ny+2,Nx+2)) # copy of C

    # Animation parameters
    labels = ["Vorticity", "Density", "Horizontal Speed", "Nutrients", "Cells"]
    fig, axs = plt.subplots(5)
    plt.tight_layout()
    for i in range(5):
        axs[i].imshow(~obstacles, cmap="gray", alpha=0.3)
        axs[i].set_title(labels[i])
        axs[i].axes.get_xaxis().set_visible(False)
        axs[i].axes.get_yaxis().set_visible(False)
    ims = []

    # Simulation Main Loop
    for it in range(Nt):
        print("\r", it, "/", Nt, end="")

        #print(obstacles)

        #simulate the flow -> extend the array
        F = np.pad(F, ((0,0),(1,1),(0,0)),'edge')
        F = np.pad(F,((0,0),(0,1),(0,0)))
        F[:,0,:] = F[:,3,:]
        F[:,Nx+1,:] = F[:,Nx-2,:]
        F[:,0,3] += Lflow

        # Drift
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

        # cut the array
        F = F[:, 1:Nx+1, :]

         #simulate the flow -> extend the array
        G = np.pad(G, ((0,0),(1,1),(0,0)),'edge')
        G = np.pad(G,((0,0),(0,1),(0,0)))
        G[:, 0 , 3] += Nflow

        # Drift Nutrients
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            G[:, :, i] = np.roll(G[:, :, i], cx, axis=1)
            G[:, :, i] = np.roll(G[:, :, i], cy, axis=0)

        # cut the array
        G = G[:, 1 : Nx + 1, :]

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
                )
            )

        F += -(dt / tau) * (F - Feq)

        # Apply collisions nutrients
        Geq = np.zeros(G.shape)
        Cl = np.sum(G, 2)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Geq[:, :, i] = Cl * w * (1 + icsc * (cx * ux + cy * uy))

        G += -(dt / taul) * (G - Geq)

        # Apply boundary
        F[obstacles, :] = bndryF
        G[obstacles, :] = bndryG


        # Calculate force density
        for x in range(Nx-1):
            for y in range(Ny-1):
                Forces[y,x] = np.array([0,0])
                for k in range (NL-1):
                    Forces[y,x] = Forces[y,x] + unary_vect[k] * g * h * (rho[y,x] - rho[y + cys[k+1], x + cxs[k+1]]) / dx[k]

        # Calculate fluid friction (another force)
        for x in range(Nx):
            for y in range(Ny):
                # Calculate speed of water out of ux (horizontal speed) and uy (vertical speed) :
                Forces[y,x] = Forces[y,x] + Kf * np.array([uy[y,x],ux[y,x]]) * rho[y,x]

        # Apply forces on cells
        C2 = np.zeros((Ny+2,Nx+2))
        for x in range(Nx-1):
            for y in range(Ny-1):
                mmvt = minmvt
                if close_obs[y+1,x+1]:
                    mmvt = minmvt_adh
                if C[y,x] > 0:
                    mvtx = 0
                    mvty = 0
                    if Forces[y,x,0] > mmvt:
                        mvty = 1
                    if Forces[y,x,0] < -mmvt:
                        mvty = -1
                    if Forces[y,x,1] > mmvt:
                        mvtx = 1
                    if Forces[y,x,1] < -mmvt:
                        mvtx = -1
                    # push 1 cell :
                    #C[y,x] = C[y,x] - 1
                    #C[y + mvty,x + mvtx] = C[y + mvty,x + mvtx] + 1
                    #push all the cells :
                    if obstacles2[y + mvty + 1,x + mvtx + 1] or (C2[y + mvty + 1,x + mvtx + 1] >= mc):
                        mvtx = -mvtx
                        mvty = -mvty
                        #if obstacles2[y + mvty + 1,x + mvtx + 1] or (C2[y + mvty + 1,x + mvtx + 1] >= mc): # I did not use it because it often freezes everything
                        #    mvtx = 0
                        #    mvty = 0
                    C2[y + mvty + 1,x + mvtx + 1] = C2[y + mvty + 1,x + mvtx + 1] + C[y,x] # ce qui est déjà arrivé dans cette case avant + la quantitée apportée


        # Separate cells when they are too many in a same square
        for x in range(1,Nx+1):
            for y in range(1,Ny+1):
                if C2[y,x] > mc: # "explosion"
                        C2[y+1,x]= C2[y+1,x] + (C2[y,x] // 4)
                        C2[y-1,x]= C2[y-1,x] + (C2[y,x] // 4)
                        C2[y,x+1]= C2[y,x+1] + (C2[y,x] // 4)
                        C2[y,x-1]= C2[y,x-1] + (C2[y,x] // 4)
                        C2[y,x] = C2[y,x] - 4*(C2[y,x] // 4)


        # Finish movement of cells
        C1 = C2[1 : Ny+1, 1 : Nx+1] # 0 [1 ... Nx] Nx+1

        # Refresh the density of water considering the cells
        for x in range(Nx-1):
            for y in range(Ny-1):
                rho[y,x] = rho[y,x] * (vol - C[y,x] * vc) / ( vol- C1[y,x] * vc )
                # if there are too many cells in the square, we assume they have a smaller volum because they are compressed, so the previous calculus does not hold and density is very high :
                if rho[y,x] < 0:
                    rho[y,x] = maxd

        # Refresh cells
        C = C1

        # Generate the GIF :

        # Plot every 10 steps
        if (it % 10) == 0:

            # vorticity
            vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (
                np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)
            )
            vorticity = np.ma.array(vorticity, mask=obstacles)
            im0 = axs[0].imshow(vorticity, cmap="bwr", label = "Vorticity", vmin = -0.3, vmax = 0.3)

            # density
            density = np.ma.array(rho, mask=obstacles)
            im1 = axs[1].imshow(density, cmap="Blues", vmin = 0, vmax = 140)

            # horizontal speed
            speed = np.ma.array(ux, mask=obstacles)
            im2 = axs[2].imshow(speed, cmap="bwr", vmin = -0.5, vmax = 0.5)

            #Nutrients
            nutri = np.ma.array(np.sum(G,2), mask = obstacles)
            im3 = axs[3].imshow(nutri, cmap="hot_r", vmin = 0, vmax = 1500)

            #Cells
            cells = np.ma.array(C, mask = obstacles)
            im4 = axs[4].imshow(cells, cmap="Greens", label = "Cells", vmin = 0, vmax = 3)

            ims.append([im0, im1, im2, im3, im4])

    # Save figure
    print("\ncreating animation")
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    print("saving animation")
    ani.save("output.gif")

    return 0

if __name__ == "__main__":
    main()
