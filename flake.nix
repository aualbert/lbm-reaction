{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs = { self, flake-utils, nixpkgs }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python310;
        packages = pkgs.python310Packages;
      in {
        devShells.default = pkgs.mkShell {
          name = "blm-reaction-dev";
          packages = [
            pkgs.ffmpeg
            python
            packages.matplotlib
            packages.numpy
            packages.imageio
            packages.tomli
	    packages.numba
          ];
        };
      });
}
