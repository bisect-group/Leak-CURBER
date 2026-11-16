#!/usr/bin/env bash
# Install pinned external command-line tools into <repo>/external/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="${EMULATOR_EXTERNAL_DIR:-$ROOT/external}"

# name | url | sha256 | unpacked dir | tar strip level | version command (run in the unpacked dir)
TOOLS=(
  "mmseqs|https://mmseqs.com/archive/76de5d8ae114750adc59288516d947b6a72a4b05/mmseqs-linux-gpu.tar.gz|c2198f4b8b49d7c77d3bc507c117e37e76e34d84a890b1189c01b8f213f6ad89|mmseqs|1|bin/mmseqs version"
  "foldseek|https://mmseqs.com/archive/463739e0014a1549a527de589102cde98f802f37/foldseek-linux-gpu.tar.gz|1786a5cb808141f40f49d0a1840592fe8af8a84c51156cbd0771ea98ab884611|foldseek|1|bin/foldseek version"
  "blast|https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.17.0/ncbi-blast-2.17.0+-x64-linux.tar.gz|3888112d8207831aa47371d93583c601f058f88b5db22dc782438b039a3a411b|ncbi-blast-2.17.0+|1|bin/blastp -version | head -1"
  "diamond-2.1.9|https://github.com/bbuchfink/diamond/releases/download/v2.1.9/diamond-linux64.tar.gz|136cc7345bc4929146fce43336170502be467598bf49222919dc38cac6af0341|diamond-2.1.9|0|./diamond version"
  "diamond-2.0.14|https://github.com/bbuchfink/diamond/releases/download/v2.0.14/diamond-linux64.tar.gz|ac36cfd1b90eddd1ecd3f12a8854d844d89e92faa4ef3425e77a0b229d933c6a|diamond-2.0.14|0|./diamond version"
)

mkdir -p "$EXT/downloads"
printf 'tool\turl\tsha256\tversion\n' > "$EXT/VERSIONS.tsv.tmp"

for row in "${TOOLS[@]}"; do
  IFS='|' read -r name url sha dir strip version_cmd <<< "$row"
  file="$EXT/downloads/$dir.tar.gz"
  if [[ "$(cat "$EXT/$dir/.installed" 2>/dev/null)" == "$sha" ]]; then
    echo "skip $name: already installed"
  else
    if [[ ! -f "$file" ]]; then
      echo "download $name"
      curl -L -C - --retry 10 --retry-all-errors -o "$file.part" "$url"
      mv "$file.part" "$file"
      if [[ "$name" == blast ]]; then
        # NCBI publishes an MD5 next to the tarball
        expected="$(curl -fsSL "$url.md5" | cut -d' ' -f1)"
        [[ "$(md5sum "$file" | cut -d' ' -f1)" == "$expected" ]] || { echo "MD5 mismatch for $file" >&2; exit 1; }
      fi
    fi
    actual="$(sha256sum "$file" | cut -d' ' -f1)"
    [[ "$actual" == "$sha" ]] || { echo "SHA256 mismatch for $file: expected $sha, got $actual" >&2; exit 1; }
    echo "extract $name"
    rm -rf "${EXT:?}/$dir" "$EXT/$dir.tmp"
    mkdir "$EXT/$dir.tmp"
    tar -xzf "$file" -C "$EXT/$dir.tmp" --strip-components="$strip"
    mv "$EXT/$dir.tmp" "$EXT/$dir"
    [[ "$name" != diamond-* ]] || chmod +x "$EXT/$dir/diamond"
    echo "$sha" > "$EXT/$dir/.installed"
  fi
  printf '%s\t%s\t%s\t%s\n' "$name" "$url" "$sha" "$(cd "$EXT/$dir" && eval "$version_cmd")" >> "$EXT/VERSIONS.tsv.tmp"
done

mv "$EXT/VERSIONS.tsv.tmp" "$EXT/VERSIONS.tsv"
cat "$EXT/VERSIONS.tsv"
