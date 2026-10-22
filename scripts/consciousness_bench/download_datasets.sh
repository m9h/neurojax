#!/usr/bin/env bash
# Download the openly licensed datasets of the consciousness complexity benchmark.
# Layout: /data/raw/consciousness-bench/<dataset>/ (override with CB_ROOT).
#
#   propofol-chennu2016        Chennu et al. 2016 PLOS Comp Biol, 20 subj x 4 levels,
#                              91-ch EEGLAB, CC BY 2.0 UK (Cambridge repository)
#   dmt-pallavicini2021        Pallavicini et al. 2021, inhaled DMT, 35 subj,
#                              DMT/EC/EO BDF (+2 XDF), CC BY 4.0 (Zenodo 3992359)
#   psilocybin-microdose-cavanna2022  Cavanna et al. 2022 Transl Psychiatry,
#                              34 subj, dose/placebo, CC BY 4.0 (Zenodo 5745892, Data.rar)
#
# NOT downloadable here (restricted, motivated access request on Zenodo):
#   Sarasso 2015 / Maschke 2024 three-drug set (propofol, xenon, ketamine, N=5 each)
#   https://zenodo.org/records/806176  -> place under $CB_ROOT/anesthesia-sarasso2015/
set -euo pipefail
CB_ROOT="${CB_ROOT:-/data/raw/consciousness-bench}"
mkdir -p "$CB_ROOT"

echo "== propofol-chennu2016"
d="$CB_ROOT/propofol-chennu2016"; mkdir -p "$d"
[ -f "$d/sedation-restingstate.zip" ] || \
  curl -L --retry 5 -C - -o "$d/sedation-restingstate.zip" \
  "https://www.repository.cam.ac.uk/bitstreams/e94a6722-da5b-4e53-8673-5e8ec106e0f7/download"
[ -d "$d/unzipped" ] || (mkdir -p "$d/unzipped" && unzip -q -o "$d/sedation-restingstate.zip" -d "$d/unzipped")

echo "== dmt-pallavicini2021"
d="$CB_ROOT/dmt-pallavicini2021"; mkdir -p "$d"
curl -sL "https://zenodo.org/api/records/3992359" | python3 -c '
import json, sys
for f in json.load(sys.stdin)["files"]:
    print(f["key"], f["links"]["self"], f["checksum"])' | while read -r key url md5; do
  [ -f "$d/$key" ] && continue
  curl -sL --retry 5 -o "$d/$key" "$url"
done
echo "   $(ls "$d" | wc -l) files"

echo "== psilocybin-microdose-cavanna2022"
d="$CB_ROOT/psilocybin-microdose-cavanna2022"; mkdir -p "$d"
[ -f "$d/Data.rar" ] || \
  curl -L --retry 5 -C - -o "$d/Data.rar" "https://zenodo.org/records/5745892/files/Data.rar?download=1"
if ! [ -d "$d/unrar" ]; then
  mkdir -p "$d/unrar"
  if command -v unrar >/dev/null; then unrar x -o+ "$d/Data.rar" "$d/unrar/" >/dev/null
  elif command -v 7z >/dev/null; then 7z x -y -o"$d/unrar" "$d/Data.rar" >/dev/null
  else echo "   no unrar/7z: extract $d/Data.rar manually"; rmdir "$d/unrar"; fi
fi
echo "done"
