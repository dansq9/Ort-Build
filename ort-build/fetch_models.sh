#!/usr/bin/env bash
# Fetches the publicly hosted release models at pinned URLs and checks size + SHA-256.
# (The third release model, Spleeter 5stems piano, is not hosted yet; its coverage is checked in the app repo.)
set -euo pipefail
out=${1:?usage: fetch_models.sh OUT_DIR}; mkdir -p "$out"
fetch() { # url file size sha256
  [[ -f "$2" ]] || { curl -fsSL --retry 3 -o "$2.part" "$1"; mv "$2.part" "$2"; }
  [[ "$(stat -c %s "$2")" == "$3" ]] || { echo "$2: wrong size" >&2; exit 1; }
  echo "$4  $2" | sha256sum -c -
}
fetch "https://huggingface.co/audiojs/scnet/resolve/e71bcdb42bce1d62b7fae25ba90fcbcf90f73f46/scnet.int8.onnx" \
  "$out/scnet.int8.onnx" 12900277 98228931494151762a1c4ab1ec7899a894b1f81fd4a509921a9d2f9bacc50845
fetch "https://raw.githubusercontent.com/spotify/basic-pitch/9991303bba609a3b93089d13ec80d1d495083596/basic_pitch/saved_models/icassp_2022/nmp.onnx" \
  "$out/nmp.onnx" 230444 2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec
