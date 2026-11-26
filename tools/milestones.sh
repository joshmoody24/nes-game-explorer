#!/usr/bin/env bash
# Milestone race: seconds until a gain from start states near acquisitions, plain vs a variant.
# Usage: tools/milestones.sh <name> "<variant flags>" <start states...>   (8 runs at a time, 3 workers each)
set -uo pipefail
name="$1"; variant="$2"; shift 2
rm -rf "runs/$name"
jobs_list=()
for start in "$@"; do
  for seed in 0 1 2; do
    for kind in plain variant; do
      jobs_list+=("$start $seed $kind")
    done
  done
done
run_one() {
  read -r start seed kind <<< "$1"
  flags=""; [ "$kind" = variant ] && flags="$variant"
  tag="$(basename "$start" .state)-$kind-s$seed"
  tools/capped 6G nix develop -c python -u go_explore.py "$name/$tag" --minutes 3 --workers 3 --seed "$seed" \
    --start-state "$start" --keep-more 4217 --stop-at-item bombs $flags > "/tmp/$name-$tag.log" 2>&1 < /dev/null
}
export -f run_one; export name variant
printf '%s\n' "${jobs_list[@]}" | xargs -P ${PARALLEL:-8} -I{} bash -c 'run_one "$@"' _ {}
for d in runs/$name/*/; do
  python3 - "$d" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines()]
first = rows[0]
gain = next((r["seconds"] for r in rows if r["energy tanks"] > first["energy tanks"] or len(r["items"]) > len(first["items"])
             or r["max missiles"] > first["max missiles"]), None)
print(f"{d.name}: {'gain at %.0f s' % gain if gain is not None else 'no gain'} ({rows[-1]['items']}, tanks {rows[-1]['energy tanks']})")
EOF
done
