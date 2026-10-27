#!/usr/bin/env bash
# Сборка нативного ядра.
#   Linux:   hybridrip/bin/libhybridrip_core.so
#   Windows: hybridrip/bin/hybridrip_core.dll  (кросс-компиляция MinGW-w64, всё статически —
#            DLL не требует ни Visual C++ Redistributable, ни других библиотек)
set -euo pipefail
cd "$(dirname "$0")"
OUT=../hybridrip/bin
mkdir -p "$OUT"
FLAGS="-O3 -std=c++17 -fopenmp -fno-math-errno -Wall -Wextra"

if command -v g++ >/dev/null; then
  g++ $FLAGS -march=x86-64-v2 -mtune=generic -shared -fPIC -fvisibility=hidden \
      hybridrip_core.cpp -o "$OUT/libhybridrip_core.so"
  echo "OK  $OUT/libhybridrip_core.so"
fi

if command -v x86_64-w64-mingw32-g++ >/dev/null; then
  x86_64-w64-mingw32-g++ $FLAGS -march=x86-64-v2 -mtune=generic -shared \
      -static -static-libgcc -static-libstdc++ \
      hybridrip_core.cpp -o "$OUT/hybridrip_core.dll" -Wl,--out-implib,/dev/null
  x86_64-w64-mingw32-strip "$OUT/hybridrip_core.dll"
  echo "OK  $OUT/hybridrip_core.dll"
fi
