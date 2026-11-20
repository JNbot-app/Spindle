#!/usr/bin/env bash
# Build the engine for the browser: demo/wasm/spindle.js + spindle.wasm. Needs Emscripten (em++).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
native="$here/.."
out="${1:-$here/../../demo/wasm}"
mkdir -p "$out"
sources=(archive math cq4 workers decoder confidence generation api kernels kernels_scalar kernels_wasm)
files=()
for name in "${sources[@]}"; do files+=("$native/$name.cpp"); done
em++ -O3 -std=c++17 -msimd128 -ffp-contract=off -fexceptions \
  -DSPINDLE_BUILD -DSPINDLE_STATIC -DSPINDLE_BLAS_NAME='"wasm"' \
  -I"$here/include" -I"$native" \
  "${files[@]}" "$here/cblas.cpp" "$here/api.cpp" \
  -o "$out/spindle.js" \
  -sMODULARIZE -sEXPORT_ES6 -sEXPORT_NAME=createSpindle -sENVIRONMENT=web,worker,node \
  -sSTACK_SIZE=8388608 -sALLOW_MEMORY_GROWTH -sINITIAL_MEMORY=268435456 -sMAXIMUM_MEMORY=2147483648 \
  -sEXPORTED_FUNCTIONS=_malloc,_free,_sp_load,_np_free,_np_reset,_np_eval,_np_set_prefix,_np_prefix_reused,_np_confidence,_np_set_window,_np_set_decode_skip,_np_abi_version \
  -sEXPORTED_RUNTIME_METHODS=FS,HEAPU8,HEAPF32,UTF8ToString
ls -l "$out"
