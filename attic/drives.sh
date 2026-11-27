#!/usr/bin/env bash
# Drive combinations from the Long Beam state: all off, all on, and each drive left out; 2 seeds each.
# Runs 6 at a time (4 workers each), 6 minutes, then prints a summary.
cd ~/code/metroid-explorer
POWER="--favor-holdings 10"
SAFETY="--danger-model cache/danger-model.pkl --avoid-danger 3"
SURPRISE="--surprise 5"
MASTERY="--mastery 5"
declare -A CONFIGS=(
  [none]=""
  [all]="$POWER $SAFETY $SURPRISE $MASTERY"
  [no-power]="$SAFETY $SURPRISE $MASTERY"
  [no-safety]="$POWER $SURPRISE $MASTERY"
  [no-surprise]="$POWER $SAFETY $MASTERY"
  [no-mastery]="$POWER $SAFETY $SURPRISE"
)
rm -rf runs/drives && mkdir -p runs/drives
names=(none all no-power no-safety no-surprise no-mastery)
for batch in 0 1; do
  for name in "${names[@]}"; do
    tools/capped 6G nix develop -c python -u go_explore.py drives/$name-s$batch --minutes 6 --workers 4 --seed $batch \
      --start-state cache/start-long-beam.state --keep-more 4217 --stop-at-item bombs ${CONFIGS[$name]} \
      > /tmp/drives-$name-$batch.log 2>&1 < /dev/null &
  done
  wait
done
echo done > /tmp/drives.done
