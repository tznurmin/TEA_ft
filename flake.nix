{
  description = "CUDA support for fine-tuning experiments with TEA";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        buildInputs = [
          pkgs.cudatoolkit
          pkgs.python38
          pkgs.zlib
        ];
      in
      {
        devShell = pkgs.mkShell {
          inherit buildInputs;

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [pkgs.cudatoolkit pkgs.stdenv.cc.cc.lib pkgs.zlib]}:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
          '';
        };
      });
}