#!/usr/bin/env bash
# Create the conda environment, install the pinned packages and this repository.
# Usage: bash scripts/setup_env.sh [-n ENV_NAME] [--tools]
#   -n ENV_NAME  environment name (default: the name in environment.yaml)
#   --tools      also install MMseqs2, Foldseek, BLAST and DIAMOND into external/
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"
ENV_NAME="$(sed -n 's/^name: *//p' environment.yaml)"
TOOLS=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) ENV_NAME="${2:?-n needs a name}"; shift 2 ;;
    --tools) TOOLS=true; shift ;;
    *) echo "usage: bash scripts/setup_env.sh [-n ENV_NAME] [--tools]" >&2; exit 2 ;;
  esac
done

for PKG in mamba micromamba conda; do
  command -v "$PKG" > /dev/null && break
  PKG=
done
[[ -n "$PKG" ]] || { echo "Install conda, mamba or micromamba first." >&2; exit 1; }

if "$PKG" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment '$ENV_NAME' already exists. Remove it or choose another name with -n." >&2
  exit 1
fi

echo "Creating environment '$ENV_NAME' with $PKG"
if [[ "$PKG" == micromamba ]]; then
  micromamba create -y -f environment.yaml -n "$ENV_NAME"
else
  "$PKG" env create -f environment.yaml -n "$ENV_NAME"
fi

RUN=("$PKG" run -n "$ENV_NAME")
[[ "$PKG" != conda ]] || RUN+=(--no-capture-output)
"${RUN[@]}" python -m pip install --no-deps -r requirements-nodeps.txt
"${RUN[@]}" python -m pip install --no-deps -e .

if [[ ! -f .env ]]; then
  sed "s|^EMULATOR_PROJECT_ROOT=.*|EMULATOR_PROJECT_ROOT=\"$ROOT\"|" .env.example > .env
  echo "Created .env with EMULATOR_PROJECT_ROOT=$ROOT"
fi

if $TOOLS; then
  bash scripts/install_external_tools.sh
fi

"${RUN[@]}" python -c "import torch, dgl, localmapper, MolR.featurizer, src, hydra_plugins; print('ok')"
echo "Done. Activate with: $PKG activate $ENV_NAME"
