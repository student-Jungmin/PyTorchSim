#!/usr/bin/env bash
# Build Gem5 / Spike from source.
#
# Versions are pinned in thirdparty/github-releases.json - the same manifest
# the CI docker image (ghcr.io/psal-postech/torchsim-ci) is built against.
# Cloning untagged HEADs has caused mlir-opt option-name drift in the past
# (e.g. test-tile-operation-graph's `sample-mode` <-> `tls_mode` rename), so
# always honor the pinned release_tag for a known-good Python<->mlir-opt pair.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${ROOT}/thirdparty/github-releases.json"
home="/workspace"

if [ ! -f "$MANIFEST" ]; then
    echo "error: pin manifest not found at $MANIFEST" >&2
    exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "jq not found, installing..."
    apt -y update && apt -y install jq
fi

read_pin() {
    # $1 = key (gem5 / llvm_project / spike), echoes "<repo> <tag>"
    jq -r --arg k "$1" '.[$k] | "\(.repository) \(.release_tag)"' "$MANIFEST"
}

read GEM5_REPO  GEM5_TAG  <<< "$(read_pin gem5)"
read SPIKE_REPO SPIKE_TAG <<< "$(read_pin spike)"

echo "Building from source using pins in $MANIFEST:"
echo "  gem5  = ${GEM5_REPO} @ ${GEM5_TAG}"
echo "  spike = ${SPIKE_REPO} @ ${SPIKE_TAG}"

cd "$home"

# Gem5
apt -y update && apt -y upgrade && apt -y install scons
git clone --depth 1 --branch "$GEM5_TAG" "https://github.com/${GEM5_REPO}.git"
cd gem5 && scons build/RISCV/gem5.opt -j "$(nproc)"
export GEM5_PATH="$home/gem5/build/RISCV/gem5.opt"
cd "$home"

# NO LLVM STEP. It built a RISC-V LLVM into /riscv-llvm with MLIR python
# bindings, for PyTorchSimFrontend/tog to parse the compiler's IR with. tog now
# lives in the compiler, which brings its own LLVM 23, and nothing in this
# repository imports mlir.
cd "$home"

# Spike Simulator
git clone --depth 1 --branch "$SPIKE_TAG" "https://github.com/${SPIKE_REPO}.git"
cd riscv-isa-sim && mkdir -p build && cd build && \
    ../configure --prefix="$RISCV" && make -j && make install
cd "$home"
