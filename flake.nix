{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = [
          (pkgs.python313.withPackages (ps: [ ps.jax ps.optax ps.numpy ps.matplotlib ]))
        ];
        QUICKNES_CORE = "${pkgs.libretro.quicknes}/lib/retroarch/cores/quicknes_libretro.so";
        FCEUMM_CORE = "${pkgs.libretro.fceumm}/lib/retroarch/cores/fceumm_libretro.so";
      };
    };
}
