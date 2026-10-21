# Reduced ONNX Runtime for Android

Builds ONNX Runtime 1.31.0 from source with only the operators a small set of audio models uses (a source
separation model and Spotify's Basic Pitch), to shrink the Android native library. It also builds a Linux wheel
and checks that its outputs match the stock runtime.

- `ort-build/build_reduced_ort.sh`: the build (pinned ORT commit, NDK and CMake versions)
- `ort-build/release_models.required_operators_and_types.config`: the operator config
- `ort-build/check_ops_covered.py`, `ort-build/parity.py`, `ort-build/fetch_models.sh`: checks

Results: the workflow summary (native library sizes, parity) and the `onnxruntime-android-reduced-1.31.0`
artifact. ONNX Runtime is MIT-licensed (Microsoft).
