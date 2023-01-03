# lbm-reaction
A Lattice Boltzmann method with reactions to model E. coli growth.
This simulation was written as part of the initiation to research course of the MPRI.

## Usage

#### Requirements
With nix installed, simply type `nix develop`. Otherwise, the following packages are needed:
- python with matplotlib, numpy, imagio, tomli and numba
- ffmpeg 

#### Running a simulation
To make a simulation one need a folder with :
- one picture describing te initial situation
- one file called config.toml

fill the config.toml file with :
- steps : the number of simulation step, one steps last 10 ms on the output, the simulation is real time
- image : an image that defines the initial simulation, the size in pixel of the picture defines the size of the output image, 400 * 200 pixels is a good order of magnitude
- flow : the water flow, nb of water elements added every 10 ms in every entering bucket
- species : list of species with each the fields : name (string) , color \[r,g,b\] out of 255 (refering to the color used in the image), inflow
- cells : list of species of cells with each the fields : name, color, inflow, size (10^-6 m)
- reactions : list of reactions with each the fields : reactants \[stoechiometric coefficient, name (refering to the species and cells names)\] , products (with the same format), rate (speed coefficient,\[list of the exponent coeficient of each species in the reaction speed formula\]
then enter the command python3 run.py path/to/config.toml`.

![output.gif](https://github.com/aualbert/lbm-reaction/blob/main/output.gif)
