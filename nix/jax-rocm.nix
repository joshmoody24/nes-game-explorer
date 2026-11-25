# JAX on AMD GPUs: AMD's jax-rocm7 plugin wheels, running on the ROCm runtime they were built for
# (ROCm 7.14, from AMD's "TheRock" wheels). Mixing the plugin with nixpkgs' ROCm 7.2.4 crashed in the
# HSA runtime's async event thread, so the whole GPU stack comes from one matching set.
# Returns { jaxRocm = ps: <python package>; rocmRoot = <path for ROCM_PATH>; }.
{ pkgs }:
let
  amdWheel = { name, hash }: pkgs.fetchurl {
    url = "https://rocm.nightlies.amd.com/v2/gfx110X-all/${name}";
    inherit hash;
  };
  # AMD's ROCm runtime (core) and math libraries for RDNA3 cards (gfx110X), merged into one tree. The
  # libraries find each other through relative paths; autoPatchelf only connects them to Nix's
  # glibc and libstdc++. Unused tools (debuggers, Python bindings) may have unmet dependencies.
  rocmRoot = pkgs.stdenv.mkDerivation {
    pname = "rocm-therock";
    version = "7.14.0a20260612";
    srcs = [
      (amdWheel {
        name = "rocm_sdk_core-7.14.0a20260612-py3-none-linux_x86_64.whl";
        hash = "sha256-tgI9SyBPJicUmDyVPgxDkGHkSbLIQUMxTwucCBP1P0c=";
      })
      (amdWheel {
        name = "rocm_sdk_libraries_gfx110x_all-7.14.0a20260612-py3-none-linux_x86_64.whl";
        hash = "sha256-eytEKkLiZhAz5yyi4o4wpq/Nln8Uamq0flNweRnI+1c=";
      })
    ];
    nativeBuildInputs = [ pkgs.unzip pkgs.autoPatchelfHook ];
    buildInputs = [ pkgs.stdenv.cc.cc.lib pkgs.zlib pkgs.zstd ];
    autoPatchelfIgnoreMissingDeps = true;
    dontStrip = true;
    unpackPhase = ''
      for src in $srcs; do unzip -q -o "$src"; done
    '';
    installPhase = ''
      mkdir -p $out
      cp -r _rocm_sdk_core/. $out/
      cp -r _rocm_sdk_libraries_gfx110X_all/lib/. $out/lib/
      rm -f $out/__init__.py
      # Where XLA looks for the device bitcode and the GPU code linker.
      ln -s lib/llvm $out/llvm
      ln -s lib/llvm/amdgcn $out/amdgcn
    '';
    preFixup = ''
      addAutoPatchelfSearchPath $out/lib
      addAutoPatchelfSearchPath $out/lib/rocm_sysdeps/lib
      addAutoPatchelfSearchPath $out/lib/llvm/lib
    '';
  };
  # AMD's JAX plugin wheels, linked against that runtime.
  pluginWheel = ps: { pname, url, hash, propagated ? [ ] }: ps.buildPythonPackage {
    inherit pname;
    version = "0.12.0";
    format = "wheel";
    src = pkgs.fetchurl { inherit url hash; };
    nativeBuildInputs = [ pkgs.autoPatchelfHook ];
    buildInputs = [ rocmRoot pkgs.stdenv.cc.cc.lib pkgs.zlib pkgs.zstd ];
    preFixup = ''
      addAutoPatchelfSearchPath ${rocmRoot}/lib
      addAutoPatchelfSearchPath ${rocmRoot}/lib/rocm_sysdeps/lib
    '';
    propagatedBuildInputs = propagated;
    dontCheckRuntimeDeps = true;
  };
  jaxRocm = ps:
    let
      pjrt = pluginWheel ps {
        pname = "jax-rocm7-pjrt";
        url = "https://files.pythonhosted.org/packages/39/f4/14bbe17e01d83ce5583dbef46ab8daad88522e63594b71598462cf1ff6cb/jax_rocm7_pjrt-0.12.0-py3-none-manylinux_2_27_x86_64.whl";
        hash = "sha256-RVHQcUUjNV8pRsoZWMP1hOlWYeRnpkHTa3JvcYoceTQ=";
      };
    in
    pluginWheel ps {
      pname = "jax-rocm7-plugin";
      url = "https://files.pythonhosted.org/packages/5c/74/c9a54915f81f9781e2934d6f7f3ea2a0fb4d635b9b436640a5d995d27900/jax_rocm7_plugin-0.12.0-cp313-cp313-manylinux_2_27_x86_64.whl";
      hash = "sha256-f7sLby/PNVt2ysNYgR2L4jX/qyian7X3rcAXVgX8hTw=";
      propagated = [ pjrt ];
    };
in
{
  inherit jaxRocm rocmRoot;
}
