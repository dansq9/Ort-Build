#!/usr/bin/env bash
# Builds a reduced-operator ONNX Runtime AAR for the release app (see README.md in this folder).
#
#   tools/ort-build/build_reduced_ort.sh [android|host|all]
#
#   android (default)  arm64-v8a + armeabi-v7a (+ ABIS="... x86_64") -> out/onnxruntime-android-reduced-<ver>.aar
#   host               Linux x64 shared lib + Python wheel with the same op config (verification only)
#
# Environment (all optional):
#   WORK          scratch dir for the source tree and build trees (default: $TMPDIR/ort-build)
#   ANDROID_HOME  Android SDK (default /home/user/android-sdk); NDK + cmake are installed into it
#   ABIS          Android ABIs to build (default "arm64-v8a armeabi-v7a")
#   KEEP_BUILD=1  keep the per-ABI build trees (default: deleted after the .so files are copied)
#   REGEN_CONFIG=1  regenerate the op config from the pinned release models (fetch_release_models.sh:
#                   SCNet int8, Basic Pitch nmp.onnx, Spleeter 5stems piano) instead of using the
#                   committed release_models.required_operators_and_types.config
#   SPLEETER5_ONNX / SPLEETER5_URL  with REGEN_CONFIG=1: where to get the not-yet-hosted Spleeter piano
#                   model (see fetch_release_models.sh; default: reproduce it from Deezer's tarball)
#
# Non-minimal (full ONNX loader) reduced-op build: the app keeps loading the pinned .onnx file and
# applies graph optimisations at runtime, so no .ort conversion or model-manifest change is needed.
set -euo pipefail

MODE="${1:-android}"
ORT_VERSION="1.31.0"
ORT_COMMIT="402979ac1bef89dcd55add41eaa7277c407599eb"   # tag v1.31.0
NDK_VERSION="28.0.13004108"                              # what ORT 1.31's own Android CI uses
CMAKE_VERSION="3.31.6"                                   # ditto
ANDROID_API=26                                           # app minSdk
BUILD_CONFIG="MinSizeRel"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/out"
OPS_CONFIG="$HERE/release_models.required_operators_and_types.config"
WORK="${WORK:-${TMPDIR:-/tmp}/ort-build}"
ANDROID_HOME="${ANDROID_HOME:-/home/user/android-sdk}"
ABIS="${ABIS:-arm64-v8a armeabi-v7a}"
SRC="$WORK/onnxruntime"
VENV="$WORK/venv"
NDK="$ANDROID_HOME/ndk/$NDK_VERSION"
CMAKE_BIN="$ANDROID_HOME/cmake/$CMAKE_VERSION/bin"
export ANDROID_HOME ANDROID_SDK_ROOT="$ANDROID_HOME" ANDROID_NDK_HOME="$NDK"
export PATH="$CMAKE_BIN:$PATH"

mkdir -p "$WORK" "$OUT"
log() { printf '\n== %s\n' "$*"; }

install_toolchain() {
  local sdkm="$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager"
  [[ -d "$NDK" ]] || { log "installing NDK $NDK_VERSION"; { yes || true; } | "$sdkm" --sdk_root="$ANDROID_HOME" "ndk;$NDK_VERSION" >/dev/null; }
  [[ -x "$CMAKE_BIN/cmake" ]] || { log "installing cmake $CMAKE_VERSION"; { yes || true; } | "$sdkm" --sdk_root="$ANDROID_HOME" "cmake;$CMAKE_VERSION" >/dev/null; }
}

fetch_source() {
  if [[ ! -d "$SRC/.git" ]]; then
    log "cloning onnxruntime v$ORT_VERSION"
    git clone --depth 1 --branch "v$ORT_VERSION" https://github.com/microsoft/onnxruntime.git "$SRC"
  fi
  [[ "$(git -C "$SRC" rev-parse HEAD)" == "$ORT_COMMIT" ]] || { echo "unexpected ORT commit" >&2; exit 1; }
  if [[ ! -x "$VENV/bin/python" ]]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q "onnxruntime==$ORT_VERSION" onnx numpy flatbuffers packaging setuptools wheel
  fi
}

regen_config() {
  # Every model a release build runs on ORT (pinned, size + SHA-256 checked), then ORT's own
  # reduced-build tooling for all of them together at every optimisation level (constant folding at
  # BASIC needs Cast/Expand/ConstantOfShape kernels; EXTENDED fuses Gelu/QuickGelu; ALL adds
  # FusedConv and NCHWc on x86) on both target platforms, merged (gen_op_config.py), then checked.
  local models=()
  mapfile -t models < <("$HERE/fetch_release_models.sh" "$WORK/models")
  [[ ${#models[@]} -gt 0 ]] || { echo "fetch_release_models.sh returned no models" >&2; exit 1; }
  "$VENV/bin/python" "$HERE/gen_op_config.py" "$OPS_CONFIG" "${models[@]}"
  "$VENV/bin/python" "$HERE/check_ops_covered.py" "$OPS_CONFIG" "${models[@]}"
}

common_flags=(
  --config "$BUILD_CONFIG" --parallel --skip_tests --cmake_generator Ninja
  --build_shared_lib --enable_lto --disable_ml_ops
  --include_ops_by_config "$OPS_CONFIG" --enable_reduced_operator_type_support
  --skip_submodule_sync --no_telemetry
)
# build.py refuses to run as root (CI containers and cloud sessions often are).
[[ "$(id -u)" == 0 ]] && common_flags+=(--allow_running_as_root)

build_android() {
  local jnilibs="$WORK/jnilibs" headers=""
  rm -rf "$jnilibs"; mkdir -p "$jnilibs"
  for abi in $ABIS; do
    local bdir="$WORK/build-android-$abi"
    log "building $abi"
    (cd "$SRC" && "$VENV/bin/python" tools/ci_build/build.py "${common_flags[@]}" \
      --build_dir "$bdir" --android --android_sdk_path "$ANDROID_HOME" --android_ndk_path "$NDK" \
      --android_abi "$abi" --android_api "$ANDROID_API" --build_java --use_xnnpack)
    mkdir -p "$jnilibs/$abi"
    cp "$bdir/$BUILD_CONFIG/libonnxruntime.so" "$bdir/$BUILD_CONFIG/libonnxruntime4j_jni.so" "$jnilibs/$abi/"
    if [[ -z "$headers" ]]; then headers="$WORK/headers"; rm -rf "$headers"; cp -r "$bdir/$BUILD_CONFIG/android/headers" "$headers"; fi
    [[ "${KEEP_BUILD:-0}" == 1 ]] || rm -rf "$bdir"
  done
  log "packaging AAR"
  local aar_build="$WORK/aar-build" publish="$WORK/aar-publish"
  rm -rf "$aar_build" "$publish"
  (cd "$SRC/java" && ./gradlew --no-daemon -b=build-android.gradle -c=settings-android.gradle \
    -DjniLibsDir="$jnilibs" -DbuildDir="$aar_build" -DheadersDir="$headers" -DpublishDir="$publish" \
    -DminSdkVer="$ANDROID_API" -DtargetSdkVer=34 -DENABLE_TRAINING_APIS=0 -DreleaseVersionSuffix= clean build publish)
  local aar; aar="$(find "$publish" -name "onnxruntime-android-$ORT_VERSION.aar" | head -1)"
  cp "$aar" "$OUT/onnxruntime-android-reduced-$ORT_VERSION.aar"
  rm -rf "$aar_build" "$publish"
  (cd "$OUT" && sha256sum "onnxruntime-android-reduced-$ORT_VERSION.aar" | tee "onnxruntime-android-reduced-$ORT_VERSION.aar.sha256")
}

build_host() {
  local bdir="$WORK/build-host"
  log "building host (Linux x64) for verification"
  (cd "$SRC" && "$VENV/bin/python" tools/ci_build/build.py "${common_flags[@]}" --build_dir "$bdir" --build_wheel)
  mkdir -p "$OUT/host"
  cp "$bdir/$BUILD_CONFIG/libonnxruntime.so"* "$OUT/host/" 2>/dev/null || true
  cp "$bdir/$BUILD_CONFIG"/dist/*.whl "$OUT/host/"
  [[ "${KEEP_BUILD:-0}" == 1 ]] || rm -rf "$bdir"
}

install_toolchain
fetch_source
[[ "${REGEN_CONFIG:-0}" == 1 ]] && regen_config
case "$MODE" in
  android) build_android ;;
  host) build_host ;;
  all) build_android; build_host ;;
  *) echo "usage: $0 [android|host|all]" >&2; exit 2 ;;
esac
log "done; outputs in $OUT"
