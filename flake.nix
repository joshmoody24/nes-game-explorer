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
      rocm = import ./nix/jax-rocm.nix { inherit pkgs; };
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = [
          (pkgs.python313.withPackages (ps: [ ps.jax ps.optax ps.numpy ps.matplotlib (rocm.jaxRocm ps) ]))
        ];
        QUICKNES_CORE = "${pkgs.libretro.quicknes}/lib/retroarch/cores/quicknes_libretro.so";
        SHIM = "${shim}/lib/libshim.so";
        FCEUMM_CORE = "${pkgs.libretro.fceumm}/lib/retroarch/cores/fceumm_libretro.so";
        ROCM_PATH = "${rocm.rocmRoot}";
        # Allocate GPU memory as needed rather than 75% per process up front, and never above 90%.
        XLA_PYTHON_CLIENT_PREALLOCATE = "false";
        XLA_PYTHON_CLIENT_MEM_FRACTION = "0.9";
        # Compiled training and evaluation code is kept on disk, saving ~4 s per model on each later run.
        JAX_COMPILATION_CACHE_DIR = "/home/josh/.cache/metroid-explorer-jax";
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS = "0.5";
      };
    };
}
