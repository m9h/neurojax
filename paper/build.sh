#!/bin/bash
# Build the WAND/NeuroJAX paper: Rnw → tex (knitr) → pdf (tectonic)
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Step 1: knitr (Rnw → tex) ==="
Rscript -e "knitr::knit('wand_neurojax.Rnw')"

echo ""
echo "=== Step 2: tectonic (tex → pdf) ==="
tectonic wand_neurojax.tex

echo ""
echo "=== Done ==="
ls -lh wand_neurojax.pdf
