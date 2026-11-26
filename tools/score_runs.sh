#!/usr/bin/env bash
# Waits for every go_explore run under runs/<group>/ to finish, then scores each by progress along a known route.
# Run as a script file so that pgrep can't match this script's own command line.
# Usage: score_runs.sh <group> <route run> <gain> <steps before>
cd ~/code/metroid-explorer
group="$1"
sleep 15
while pgrep -f "^[^ ]*python[^ ]* -u go_explore.py $group/" > /dev/null; do sleep 10; done  # anchored: a shell whose command text mentions the run never matches
for d in runs/$group/*/; do
  r=${d#runs/}; r=${r%/}
  (echo "$r: $(tools/capped 10G nix develop -c python -u -m checks.autopsy_check "$r" "$2" "$3" "$4" 2>&1 | grep -oE "location [0-9]+/[0-9]+" | head -1); tanks $(tail -1 "$d/metrics.jsonl" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['energy tanks'])")") &
done
wait
