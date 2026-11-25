{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
      shim = pkgs.stdenv.mkDerivation {
        name = "libretro-shim";
        src = ./shim;
        buildPhase = "$CC -O2 -shared -fPIC shim.c -o libshim.so";
        installPhase = "mkdir -p $out/lib && cp libshim.so $out/lib/";
      };
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = [
          (pkgs.python313.withPackages (ps: [ ps.jax ps.optax ps.numpy ps.matplotlib ]))
        ];
        QUICKNES_CORE = "${pkgs.libretro.quicknes}/lib/retroarch/cores/quicknes_libretro.so";
        SHIM = "${shim}/lib/libshim.so";
        FCEUMM_CORE = "${pkgs.libretro.fceumm}/lib/retroarch/cores/fceumm_libretro.so";
      };
    };
}
