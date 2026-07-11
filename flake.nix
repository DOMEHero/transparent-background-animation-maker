{
  description = "Development environment for transparent-background-animation-maker";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          linuxLibraries = with pkgs; [
            glib
            libglvnd
            stdenv.cc.cc.lib
            zlib
          ];
          nixLibraryPath = pkgs.lib.makeLibraryPath linuxLibraries;
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              ffmpeg
              python311
              uv
            ];

            env = {
              UV_PYTHON = "${pkgs.python311}/bin/python";
              UV_PYTHON_DOWNLOADS = "never";
            };

            shellHook = ''
              ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
                # PyTorch's Linux wheel includes the CUDA runtime, but libcuda.so
                # must come from the host NVIDIA driver. NixOS exposes it under
                # /run/opengl-driver; the other entries cover common non-NixOS,
                # WSL, and container installations.
                cudaDriverPaths=(
                  /run/opengl-driver/lib
                  /run/opengl-driver-32/lib
                  /usr/lib/wsl/lib
                  /usr/lib/x86_64-linux-gnu
                  /usr/lib/aarch64-linux-gnu
                  /usr/lib64
                  /usr/local/cuda/compat
                  /usr/local/nvidia/lib64
                )
                cudaDriverLibraryPath=""

                for path in "''${cudaDriverPaths[@]}"; do
                  if [ -e "$path/libcuda.so.1" ] || [ -e "$path/libcuda.so" ]; then
                    cudaDriverLibraryPath="''${cudaDriverLibraryPath:+$cudaDriverLibraryPath:}$path"
                  fi
                done

                export LD_LIBRARY_PATH="''${cudaDriverLibraryPath:+$cudaDriverLibraryPath:}${nixLibraryPath}:''${LD_LIBRARY_PATH:-}"

                if [ -z "$cudaDriverLibraryPath" ]; then
                  echo "Warning: libcuda.so was not found. Install/configure the host NVIDIA driver before using CUDA."
                fi
              ''}
              echo "Run 'uv sync --dev' to install the project dependencies."
              echo "Check CUDA with: uv run python -c 'import torch; print(torch.cuda.is_available())'"
            '';
          };
        }
      );

      formatter = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        pkgs.nixfmt-rfc-style
      );
    };
}
