{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    systems = {
      url = "github:nix-systems/default";
    };
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
  };

  outputs = {nixpkgs, ...} @ inputs:
    inputs.flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];
      perSystem = {system, ...}: let
        pkgs = import nixpkgs {inherit system;};
      in {
        devShells.default = pkgs.mkShell {
          venvDir = ".venv";
          postShellHook = ''pip install -r requirements.txt'';
          strictDeps = false;
          packages = with pkgs.python312Packages;
            [
              # (opencv4.override {
              #   enableGtk2 = true;
              # })
              opencv4
              torch-bin
              torchvision-bin
              pydantic
              pandas
              scikit-learn
              tqdm
              timm
              albumentations
              pytorch-lightning
              torchinfo
              wandb
            ]
            ++ (with pkgs; [v4l-utils ffmpeg-full]);
        };
      };
    };
}
