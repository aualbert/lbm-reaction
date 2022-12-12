import argparse
import os.path
import tomli

from src.main import run_simulation
from src.load_config import import_image, import_reactions

# parse command line arguments
parser = argparse.ArgumentParser(
    description="Lattice Boltzmann method with reactions to model E. coli growth",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("config", help=".toml simulation config")
args = parser.parse_args()

# parse configuration file
with open(args.config, mode="rb") as fp:
    config = {
        "steps": 500,
        "Nx": 0,
        "Ny": 0,
        "species": [],
        "cells": [],
        "reactions": [],
    }
    config = config | tomli.load(fp)

    dir_path = os.path.dirname(args.config)
    image_path = os.path.join(dir_path, config["image"])
    save_path = os.path.join(dir_path, "output.gif")

    (Nx, Ny, obstacles, species, cells) = import_image(
        image_path, config["species"], config["cells"], config["Nx"], config["Ny"]
    )
    
    reactions = import_reactions (config["reactions"], config["species"], config["cells"])

    run_simulation(
        Nx,
        Ny,
        config["steps"],
        obstacles,
        species,
        config["species"],
        cells,
        config["cells"],
        save_path,
        reactions
    )
