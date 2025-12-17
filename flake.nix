{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      pkgs = import nixpkgs {
        config.allowUnfree = true;
        system = "x86_64-linux";
      };
      nvidiaDrivers = (pkgs.linuxPackages_5_15.nvidia_x11.override { }).overrideAttrs
        (oldAttrs: rec {
          pname = "nvidia";
          version = "535.161.08";
          name = "nvidia-x11-${version}-nixGL";
          src = builtins.fetchurl {
            url = "https://us.download.nvidia.com/tesla/${version}/NVIDIA-Linux-x86_64-${version}.run";
            sha256 = "sha256:0p2napgrhycyqdb8wnhcryqjwh4yarakr41h712vvh31c4p6q0hg";
          };
          useGLVND = true;
          nativeBuildInputs = oldAttrs.nativeBuildInputs or [] ++ [pkgs.zstd];
        });
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell rec {
        buildInputs = with pkgs; [
          python3
          pyright
          nvidiaDrivers
          cudatoolkit
          cudaPackages.cudnn
          ninja
        ];
        shellHook = with pkgs; ''
          export CUDA_HOME=${cudatoolkit}
          export CUDNN_HOME=${cudaPackages.cudnn}
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
          export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"
          export PATH=${cudatoolkit}/bin:$PATH
        '';
      };
    };
}

