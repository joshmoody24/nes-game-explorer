#!/usr/bin/env bash
# Prints a line whenever a run's progress gains an item, tank, missile capacity, boss or area; exits after N new
# lines or when the run ends. Usage: watch_gains.sh <run> <lines>
cd ~/code/metroid-explorer
run="$1"; want="$2"; shown=0; last=""
while [ "$shown" -lt "$want" ]; do
  line=$(tail -1 "runs/$run/metrics.jsonl" 2>/dev/null | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(sorted(d['items']), d['energy tanks'], d['max missiles'], d['bosses'], sorted(d['areas']))" 2>/dev/null)
  if [ -n "$line" ] && [ "$line" != "$last" ]; then
    secs=$(tail -1 "runs/$run/metrics.jsonl" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(round(d['seconds']/60,1), 'min,', d['rooms'], 'rooms,', d['places'], 'places')")
    [ -n "$last" ] && { echo "$secs: $line"; shown=$((shown + 1)); } || echo "start: $line"
    last="$line"
  fi
  pgrep -f "^[^ ]*python[^ ]* -u go_explore.py $run" > /dev/null || { echo "run ended"; break; }
  sleep 20
done
