# lbm-reaction
A Lattice Boltzmann method with reactions to model E. coli growth.
This simulation was written as part of the initiation to research course of the MPRI.
The alternative version from Roxane can be found here: https://github.com/aualbert/lbm-reaction/tree/Method-2.

## Usage

#### Requirements
With nix installed, simply type `nix develop`. Otherwise, the following packages are needed:
- python with matplotlib, numpy, imagio, tomli and numba
- ffmpeg 

#### Running a simulation
To configure a simulation requires:
- a picture describing the initial situation
- a configuration file in the toml format

The configuration file contains:
- steps : the number of simulation step, one steps last 10 ms on the output, the simulation is real time
- image : the path to an image that defines the initial simulation, the size in pixel of the picture defines the size of the output image, 400 * 200 pixels is a good order of magnitude
- flow : the water flow, ie the number of water elements added every 10 ms in every entering bucket
- species : a list of species. A species is defined by its name, color and inflow (same as flow but for a specie). Eg. `{ name = "N", color = [204, 255, 153], inflow = 0.1 }`
- cells : a list of different types of cells. A cell is defined by its name, color and inflow. Eg. `{ name = "C", color = [204, 102, 255], inflow = 0}`
- reactions : a list of reactions. A reaction is defined by its reactants, products and rate. Eg. `{reactants = [[1,"C"], [1,"D"]], products = [], rate = [3,[2,2]]}`

To run the simulation, type `python3 run.py path/to/config`.

![output.gif](https://github.com/aualbert/lbm-reaction/blob/main/examples/exampleConcurrency/output.gif)
