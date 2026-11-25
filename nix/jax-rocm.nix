# JAX on AMD GPUs: AMD's prebuilt ROCm plugin wheels, linked against nixpkgs' ROCm.
# Returns { jaxRocm = ps: <python package>; rocmRoot = <path for ROCM_PATH>; }.
{ pkgs }:
let
  rocm = pkgs.rocmPackages;
  # AMD's wheels link to their own builds of libdrm and libnuma, renamed and with symbol versions
  # prefixed by AMDROCM_SYSDEPS_1.0 so they can't clash with system copies. Rebuild nixpkgs' sources
  # the same way.
  sysdepsVersionScript = pkgs.writeText "amdrocm-sysdeps.map" "AMDROCM_SYSDEPS_1.0 { global: *; };";
  drmSysdeps = pkgs.libdrm.overrideAttrs (old: {
    NIX_LDFLAGS = "--version-script=${sysdepsVersionScript}";
  });
  numaSysdeps = pkgs.numactl.overrideAttrs (old: {
    postPatch = (old.postPatch or "") + ''
      grep -rl 'libnuma_1\.' --include='*.c' --include='*.h' --include=versions.ldscript . | xargs sed -i 's/libnuma_1\./AMDROCM_SYSDEPS_1.0_libnuma_1./g'
    '';
  });
  rocmSysdeps = pkgs.runCommand "rocm-sysdeps" { nativeBuildInputs = [ pkgs.patchelf ]; } ''
    mkdir -p $out/lib
    cp -L ${drmSysdeps.out}/lib/libdrm.so.2 $out/lib/librocm_sysdeps_drm.so.2
    cp -L ${drmSysdeps.out}/lib/libdrm_amdgpu.so.1 $out/lib/librocm_sysdeps_drm_amdgpu.so.1
    cp -L ${numaSysdeps}/lib/libnuma.so.1 $out/lib/librocm_sysdeps_numa.so.1
    chmod u+w $out/lib/*
    patchelf --set-soname librocm_sysdeps_drm.so.2 $out/lib/librocm_sysdeps_drm.so.2
    patchelf --set-soname librocm_sysdeps_drm_amdgpu.so.1 --replace-needed libdrm.so.2 librocm_sysdeps_drm.so.2 \
      --add-rpath $out/lib $out/lib/librocm_sysdeps_drm_amdgpu.so.1
    patchelf --set-soname librocm_sysdeps_numa.so.1 $out/lib/librocm_sysdeps_numa.so.1
  '';
  rocmLibraries = (with rocm; [
    clr rocm-runtime rocblas hipblas hipblaslt miopen rccl hipsparse hipsparselt hipsolver rocsolver
    hipfft rocfft roctracer rocprofiler-register rocprofiler-sdk amdsmi
  ]) ++ [ rocmSysdeps ];
  # AMD's prebuilt wheels, with their library links rewired to nixpkgs' ROCm.
  wheel = ps: { pname, url, hash, propagated ? [ ] }: ps.buildPythonPackage {
    inherit pname;
    version = "0.12.0";
    format = "wheel";
    src = pkgs.fetchurl { inherit url hash; };
    nativeBuildInputs = [ pkgs.autoPatchelfHook ];
    buildInputs = rocmLibraries ++ [ pkgs.stdenv.cc.cc.lib pkgs.zlib pkgs.zstd pkgs.numactl ];
    propagatedBuildInputs = propagated;
    dontCheckRuntimeDeps = true;
  };
  jaxRocm = ps:
    let
      pjrt = wheel ps {
        pname = "jax-rocm7-pjrt";
        url = "https://files.pythonhosted.org/packages/39/f4/14bbe17e01d83ce5583dbef46ab8daad88522e63594b71598462cf1ff6cb/jax_rocm7_pjrt-0.12.0-py3-none-manylinux_2_27_x86_64.whl";
        hash = "sha256-RVHQcUUjNV8pRsoZWMP1hOlWYeRnpkHTa3JvcYoceTQ=";
      };
    in
    wheel ps {
      pname = "jax-rocm7-plugin";
      url = "https://files.pythonhosted.org/packages/5c/74/c9a54915f81f9781e2934d6f7f3ea2a0fb4d635b9b436640a5d995d27900/jax_rocm7_plugin-0.12.0-cp313-cp313-manylinux_2_27_x86_64.whl";
      hash = "sha256-f7sLby/PNVt2ysNYgR2L4jX/qyian7X3rcAXVgX8hTw=";
      propagated = [ pjrt ];
    };
  # XLA compiles GPU kernels at runtime and looks for ROCm's device libraries and linker under one root.
  rocmRoot = pkgs.symlinkJoin {
    name = "rocm-root";
    paths = rocmLibraries ++ [ rocm.rocm-device-libs rocm.llvm.lld ];
  };
in
{
  inherit jaxRocm rocmRoot;
}
