cd /opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/torchcodec
#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] === TorchCodec build script (final fix) ==="

# --- Step 1. Locate Torch CMake path ---
Torch_DIR=$(python -c "import torch, os; print(os.path.join(torch.__path__[0],'share','cmake','Torch'))")
echo "[INFO] Torch_DIR = $Torch_DIR"

# --- Step 2. Clean build directory ---
rm -rf build
mkdir -p build && cd build

# --- Step 3. Run CMake configure ---
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=$(which python) \
  -DTorch_DIR=$Torch_DIR \
  -DCMAKE_DISABLE_FIND_PACKAGE_NVTX=TRUE \
  -DCMAKE_DISABLE_FIND_PACKAGE_nvtx3=TRUE \
  -DTORCH_USE_NVTX=OFF \
  -DUSE_NVTX=OFF

# --- Step 4. Build shared libs ---
make -j"$(nproc)"

# --- Step 5. Copy built .so files into src ---
echo "[INFO] Copying built .so files into src/torchcodec/_core/"
cd ..
mkdir -p src/torchcodec/_core
cp build/src/torchcodec/_core/*.so src/torchcodec/_core/ || true

# --- Step 6. Install in editable mode (skip rebuild) ---
echo "[INFO] Installing torchcodec in editable mode..."
pip install -e . --no-build-isolation --no-deps

echo "[INFO] === TorchCodec build + install complete ==="
