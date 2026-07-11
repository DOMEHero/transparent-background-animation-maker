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
                export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath linuxLibraries}:''${LD_LIBRARY_PATH:-}"
              ''}
              echo "Run 'uv sync --dev' to install the project dependencies."
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
