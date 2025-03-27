{
  description = "Environment for TEA fine-tuning experiments";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          transformers
          datasets
          torchWithCuda
          evaluate
          seqeval
          accelerate
        ]);

      in {
        devShell = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudatoolkit
          ];
          shellHook = ''
            echo "READY."
          '';
        };
      });
}
