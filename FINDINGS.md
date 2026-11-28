# Findings

Dated results, newest last. Each says what was measured and how to rerun it.

## 2026-11-24: emulator throughput (unit: Ryzen 9 9900X, 12 cores / 24 threads)

Measured with `python -m checks.benchmark $QUICKNES_CORE 4` (random buttons, 4 frames per action, in-game).

| setup | 1 process | best total |
|---|---|---|
| FCEUmm core, Python callbacks | 5.5K frames/s | not measured |
| QuickNES core, Python callbacks | 26K | 349K (18 processes) |
| QuickNES, callbacks answered in C (`shim/shim.c`) | 28K | 378K |
| QuickNES, C callbacks, core told video/audio are off | 56K | **752K (18 processes)** |

- Python callbacks were only ~6% of the time; the core's own per-frame pixel conversion was ~35%.
- Telling the core video is off (`RETRO_ENVIRONMENT_GET_AUDIO_VIDEO_ENABLE`) does not change game
  behavior: RAM identical at 334 checkpoints over 20K frames of random play (`python -m checks.check_video_off`).
  Only checked near the start of the game so far.
- Remaining profile: 75% 6502 CPU emulation, ~12% audio synthesis the core still does.
- Save/load state: ~1-5 us, ~21 KB per state.

## 2026-11-24: Metroid memory and progress measures

- QuickNES exposes only the 2 KB system RAM, but Metroid keeps items, missiles, tanks and the
  Kraid/Ridley statue flags in cartridge RAM ($6000-$7FFF). That RAM is in the save state's `SRAM`
  block; `Emulator.memory()` returns system + cartridge RAM (10 KB) from a save state in ~2 us.
- Verified by experiment: $001E goes 8 -> 3 when play starts; $0050 (map x) changes 3 -> 2 walking
  into the room to the left; $6878 becomes $10 (Morph Ball) when Samus touches the pedestal
  (`python -m checks.find_morph_ball`, found by random search in 53 attempts). Other addresses in `progress.py`
  are from the Data Crystal RAM map and the nmikstas disassembly, not yet verified.
- "One-way bits" is too noisy to be a headline measure: 300 random runs from one start state flag
  ~9,800 bits, because everything the game writes once after starting looks one-way. It's kept as a
  diagnostic; watch whether it shrinks as exploration revisits places. This is also an early warning
  that the toy's sticky-bit trick may not transfer directly.

## 2026-11-24: Go-Explore with hand-made places (sanity baseline)

`python go_explore.py <name> --minutes 2` (18 workers; place = area, map room, Samus's 32-pixel block in
the room, items, tanks, max missiles / 5).

- Ending segments whenever $001E != 3 silently discarded every item pickup: collecting the Morph Ball
  switches $001E to 9 (wait timer) during the fanfare. Segments now end only on game over ($001E = 4).
  With that fix: Morph Ball within 10 s, 44 rooms after 110 s (stuck at 19 rooms from 50-90 s first).
- Throughput inside Go-Explore: ~470K frames/s (vs ~750K raw emulation), so the explorer loop costs ~35%.
- "Background layouts seen" keeps rising while rooms are flat (scrolling within a room makes new
  layouts), so it can't distinguish a stall from progress; the Metroid room count can.
- Checkpoints: ~2 KB per archived place compressed.

## 2026-11-24: 30-minute hand-made-places baseline (`runs/baseline-30min`)

18 workers, 855M frames (~478K frames/s), 8,551 places.

| time | rooms | notes |
|---|---|---|
| 10 s | 10 | Morph Ball |
| 70-80 s | 46-59 | first Missile Tank (max missiles 5); Norfair (verified by screenshots) |
| 130 s | 77 | |
| 6-10 min | 93 | plateau |
| 12 min | 131 | |
| 18-26 min | 143 | plateau |
| 30 min | 151 | still only Morph Ball + 5 missiles; no Bombs, energy tanks or bosses |

- Likely the red door that needs 5 missiles (the same wall Antithesis describes in
  https://antithesis.com/blog/2025/metroid/): places count max missiles, not current missiles, so
  "saved 5 missiles at the door" and "spent them" are the same place. Not yet confirmed.
- Norfair states in the archive show EN 00-01: places ignore energy, so near-dead states are kept.

## 2026-11-24: speed levers, round 1

Races: `python race.py <name> <minutes> "<label>=<flags>" ...` (3 seeds x 6 workers each, run side by side).
2.5 minutes at 6 workers is ~50 s of whole-machine time, so these races only cover the early game.

- **Correction to the throughput table above:** `checks.benchmark` presses random buttons including
  START, so the game is often paused, which is cheap to emulate. Real in-game emulation is ~39-44K
  frames/s per process, ~480-500K across the machine, which Go-Explore already reaches (~478K).
- Worker bookkeeping is now ~3-6% of worker time (`python -m checks.profile_worker`), after reading
  place bytes through `Emulator.view()` instead of building the 10 KB memory array each step.
  Raw speed is at its ceiling; further gains must come from wasting fewer frames.
- Ending segments after 20 steps without a new place is much worse (5 vs 16 rooms in 1.5 min): 10% of
  new places need 50+ steps of effort since the previous one, 1% need 94+ (measured over a 90 s run).
- 300-step segments are slightly worse than 100 (15-17 vs 17-19 rooms, 880-970 vs 1000-1080 places,
  all 3 seeds agreeing).

## 2026-11-24: the "red door wall" was two bugs and one real problem

- **Bug: SELECT was left out of the button set** ("it just switches weapons"), so missiles could never
  be fired: all 8,558 states of the first 30-min run had the beam selected and 0 or 5 missiles. Fixed;
  the only excluded button is now START. Lesson: never prune actions with game knowledge.
- **Bug: the room count double-counted Brinstar.** The area byte reads $00 or $10 for Brinstar; 57 of
  139 "rooms" in one checkpoint were duplicates (81 real map cells). Every room count before this entry
  is inflated ~1.7x. Areas now use the low 4 bits, in both progress and hand-made places.
- **Real problem: missiles get spent before they're needed.** 30-min run with SELECT
  (`runs/baseline-select`, 895M frames, 12,261 places): reached 10 max missiles (vs 5) but no Bombs,
  tanks or bosses. Its archive shows missiles fired often but almost always to empty: (max 5, 0 left)
  3,732 states vs (5, 5) 59; (10, 0) 1,346 vs at most 5 left with max 10. A place keeps its first state,
  and "here with missiles" equals "here without", so red doors are reached empty-handed. This is the
  wall Antithesis fixed by hand ("prefer more missiles"); we need a generic way to notice resources.
- The stall rule (archive growth under 1% in 10 min) never fired: places grew 57% in the last 10 min
  while items stayed flat. New places are cheap; it measures wandering, not progress.

## 2026-11-24: keeping states by learned promise or by empowerment

- **Learned promise (`promise.py`, `python -m checks.promise_check`)**: a model from raw memory to
  "new places found per trip", trained on a 4-minute run resumed from the stuck archive. Only 231
  discoveries in 411,775 trips, so too little signal: 23% deviance explained on held-out places (95%
  on training), refilling missiles *lowered* predicted promise, and the top bytes looked like room
  and enemy data. Outcome-based promise can't learn the value of something that has never paid off.
- **Hand-given "keep the state with more missiles"** (resumed from the stuck archive, 2.5 min, 3 seeds):
  no difference from keeping the first state.
- **Empowerment (`empowerment.py`)**: replay random buttons from one save state; because emulation is
  deterministic, memory bits that differ between replays were changed by the buttons. Paired test
  (`python -m checks.empowerment_check`): the same state edited from 0 to 5 missiles scores higher in
  93% of 100 states (8 replays x 30 steps); flipping a never-varying byte changes it in 0%. Cheaper
  settings: 4 x 10 steps still 81% higher / 10% lower; 2 x 15 drops to 55% / 21%.
- **Keeping states by empowerment** (fresh 15-min run, `runs/empowerment-15min`, 4 x 10 replays per
  scored state; new places plus one sampled revisit per segment scored): exploration throughput fell
  to ~330K frames/s, yet milestones came 1.5-1.8x sooner than the keep-first 30-min run (first
  missiles 2:24 vs 4:18, Norfair 3:10 vs 4:46), and it collected the first energy tank of any run
  (14.7 min, Brinstar map 7,25, without the Long Beam). But it did not keep ammo: 95% of states with
  missile capacity are empty, the same as keep-first (93%). Likely because scores vary ~4x between
  states while ammo moves them ~6%, so unpaired comparisons ignore ammo; and random play spends
  missiles within seconds, so full-ammo revisits are rare. One run each; not yet repeated with seeds.
- The walkthrough route (Long Beam, then lava corridors with an energy tank) is an order, not a
  requirement: the energy tank area was reached without the Long Beam.

## 2026-11-24: rarity-weighted empowerment, and why "rare" isn't enough

- On 100 realistic pairs (different states for the same place from two runs, one edited to 5 missiles;
  `python -m checks.rare_control_check baseline-select empowerment-15min`), picking the armed state:
  plain empowerment total 65%, rarity-weighted "what only I can change" 70%, rarity-weighted total per
  state 70%. Paired (same state) the plain total got 93%: most of the signal is lost to differences
  between states.
- Fresh 15-min run keeping states by rarity-weighted empowerment (`runs/rare-empowerment-15min`): early
  milestones at the same pace as plain empowerment, second Missile Tank at 3:20 (earliest of any run),
  final 82 rooms, Morph Ball + 10 missiles, no Long Beam / tanks / Bombs. Ammo kept no better: states
  with missile capacity and any ammo left: keep-first 7%, empowerment 6%, rare-empowerment 7%.
- Ranking bytes by how often random replays can change them (400 states, 8 x 30): ~2,500 bytes are
  changeable in 0.5-30% of states. Current missiles ($6879, 3.5%) sits among enemy tables ($04xx-$05xx),
  room data ($6Bxx), max missiles ($687A) and tanks ($0107). Rarity alone can't single out resources,
  so adding "rare" bytes to places would explode the archive.
- Supply looks like the real limit: random play spends missiles within seconds, so full-ammo states are
  ~1-2% of states with capacity in every run, whatever the keep rule.

## 2026-11-24: vitals, safety, and the limits of short random probes

- **Resource proxy** ("my buttons change it, it holds still when I do nothing";
  `python -m checks.resource_vital_check`): cuts 2,497 sometimes-changeable bytes to 158; current
  missiles, weapon and max missiles pass, items barely miss (80% still), plus ~150 others (mostly $62xx-$63xx).
- **Generic danger** (share of random-play replays that end with buttons changing < 20 bits): agrees
  with real game over in 84% of states. It exposed that archives are mostly near death: 387 of 400
  sampled states had energy 0-9. Final archives, share of states at energy 0-2: keep-first 95%,
  empowerment 82%, rare-empowerment 69%, safety keep 66%, safety + powerful selection 71%.
- **Keeping the safest state per place** (`runs/safety-15min`): only modestly healthier by the end
  (its 5-min checkpoint looked much better, median energy 6, but that was early-archive bias).
  Supply again: healthy states rarely arrive to be kept.
- **Favoring stronger states when choosing where to explore** (`--selection powerful`,
  `runs/safety-powerful-15min`): 1.5-2.5 min behind on every milestone, 89 rooms, no healthier. A bias
  toward safe states pulls exploration away from the frontier. Failed lever.
- **Counterfactual edits with short random replays** (`python -m checks.counterfactual_check`): nearly
  blind. Editing current missiles changes empowerment by 0.4%, health by <1%; weapon 7%, items 4%. Of
  383 proxy-flagged bytes, 1% have an effect over 5%. Short random play rarely fires missiles or dies,
  so it can't reveal what only matters over longer, purposeful play.
- **Learned danger from raw memory** (`vitals.py`, `python -m checks.danger_model_check`): trained on
  4,000 states labeled by random play (lost control in 53% of replays). Held-out AUC 0.73. Raising
  health from <=5 to 29 lowered predicted danger in 100% of 2,827 states (median -0.03); editing 200
  random other bytes changed it by 0.00, none as much as healing. Deaths are common, so this signal is
  dense enough to learn "health matters" from random play; the effect is consistent but small.

## 2026-11-24: resources are worth what the player can do with them

Family tree of a 5-min keep-first run (`runs/family-5min`, 4,953 places; parents now recorded):
depth up to 174, 63% of places have no descendants. Comparing places created at similar times,
healthy places (energy 10+) have *fewer* descendants (log-descendants -0.49 vs others), near-dead places
average, places with ammo slightly fewer (-0.08). For a random-button explorer, health and ammo barely
help discovery: new territory is found by lineages that push into danger and take damage, and ammo's
one use (red doors) is almost never achieved by random play. This explains why every keep/selection rule
based on power (empowerment, rarity, safety) failed to change progress: the explorer can't use the
power. Resources gain value only for a player that can act deliberately.

## 2026-11-24: correction - health causally expands reach (the family-tree result was confounded)

`python -m checks.health_reach_check family-5min 100 30`: 100 recently discovered (frontier) states
with energy <= 9; each copied with energy edited to 29; 30 random 100-step segments from both, same
seeds. New places reached per state: 0.03 low vs 1.26 healed; healed reached more in 21% of states,
fewer in 0%. Segments ending in game over: 70% low vs 19% healed. The previous entry's "healthy places
have fewer descendants" reflected *where* healthy states are (safe, explored areas), not what health
does. Frontier states arrive nearly dead, so most exploration from the frontier dies: health supply at
the frontier is a bottleneck.

## 2026-11-24: health supply fixed, progress not; recordings for learning

- **Health bands in places + keep the healthiest** (hand-given diagnostic: energy bands 0-4 / 5-14 /
  15-29 / 30+; `runs/health-buckets-15min`): the archive became healthy (median energy 14, 72% of
  states at 10+, 1% at 0-2, vs 1 / 5% / 66% for the safety keep rule), with ~3.5x as many places
  (34,986). Progress unchanged: 97 rooms, Morph Ball, 10 missiles, one energy tank at 13.3 min, no
  Long Beam. Healthy frontier states are necessary but not sufficient: the red door still needs a
  coordinated sequence random play doesn't produce.
- **Recordings** (`go_explore --record FRACTION`, default 0.1): a 30-s run saved 4,410 segments
  (~87 steps each). `python -m checks.replay_check`: 200 of 200 sampled recordings match their
  start-state checksum and replay to identical memory twice.

## 2026-11-25: first learned player (hindsight goal reaching)

`python -m checks.policy_trips_check policy-data 4000 300` (run under `tools/capped 16G`). Data: a 5-min
run recording 20% of segments (96,590 recordings, 7.5M steps); trained on 4,000 recordings (312K steps,
~3,080 varying bytes), 5,000 updates. Test A: 164 short trips (goal 10-30 steps ahead on held-out
recordings, 2x that many steps, 25% sticky actions); places used only to judge.

- **Shortcut found and removed.** First version: loss 1.42 (3.22 = guessing), heavy use of the goal
  (swapping goals raised loss to 3.8), yet no better than random at trips (52% vs 45%; closed 53-59% of
  distance vs random's 63%). Cause: $0014 (the game's controller state) reveals the button just pressed
  100% of the time, so the model copied the goal's held button. `experience.input_echo_bytes` now drops,
  from data alone, any byte that predicts the pressed button >= 90% of the time (here exactly one).
- After the fix: loss 2.22 train / 2.44 held out (generalizes). Trips reached: learned (sampled) 60%,
  closing 75% of distance; most-likely-button 46%; random 45% (63%); exact replay of the recorded
  buttons 92% (94%). Short trips barely suffer from sticky noise when replayed.
- Memory: a data loader that indexed an npz inside a loop hit 52 GB (each slice kept a full copy).
  Fixed; runs now use `tools/capped` (systemd scope with MemoryMax).
- Timing: building data 10K steps/s (replay + 10 KB memory copy per step); training 63 updates/s on CPU
  (batch 512, ~9,240 inputs); one learned decision ~360 us vs ~80 us per emulator step.

## 2026-11-25: JAX on the AMD GPU (RX 7900 XT) via Nix

`nix/jax-rocm.nix`: AMD's `jax-rocm7-pjrt` / `jax-rocm7-plugin` 0.12.0 wheels from PyPI, autopatchelf'd
against nixpkgs' ROCm 7.2.4. The wheels also need AMD's renamed copies of libdrm, libdrm_amdgpu and
libnuma (`librocm_sysdeps_*`) whose symbols carry `AMDROCM_SYSDEPS_1.0` version labels; nixpkgs' libdrm
and numactl are rebuilt with a version script / renamed labels and renamed sonames to match. nixpkgs
packages only the CUDA plugins. Devshell sets ROCM_PATH, no preallocation, max 90% VRAM.
`python -m checks.gpu_check`: 4096^2 fp32 matmul 6.1 TFLOP/s on the GPU vs 1.9 on the CPU.

- For comparison, nixpkgs' prebuilt `torchWithRocm` (5.9 GB download) trained the same network shape
  at 767 updates/s (batch 512).
- Policy training: copying each batch from the host gave only 123 updates/s; keeping the dataset on the
  device and sampling there inside compiled `lax.scan` chunks gives 311 updates/s (vs 63 on CPU),
  including compilation. Single decisions run on the CPU (166 us jitted); on the GPU one took 806 us.
- Known plugin bugs: compiled loops of ~300+ updates segfault inside `GpuExecutable::ExecuteThunksImpl`
  (100 per call works; disabling command buffers only moves the limit), and every JAX process
  segfaults at interpreter exit (exit code 139), even after a bare `import jax`; results are unaffected.
- New bottleneck: building training data by replay, ~9,500 steps/s in one process.

## 2026-11-25: the GPU crashes were a ROCm version mismatch

AMD's `jax-rocm7` 0.11.2-0.12.1 plugins are built for ROCm 7.14.x (PyPI summary: "ROCm:7.14.0"); nixpkgs
has ROCm 7.2.4. On nixpkgs' ROCm, core dumps showed the mid-run crashes in
`rocr::core::Runtime::AsyncEventsLoop` (libhsa-runtime64) and the exit crash in an `amd::smi` destructor
during exit handlers. `nix/jax-rocm.nix` now runs the plugin on AMD's own ROCm 7.14.0 nightly wheels
(`rocm_sdk_core` + `rocm_sdk_libraries_gfx110x_all` 7.14.0a20260612 from rocm.nightlies.amd.com/v2/
gfx110X-all, ~1.4 GB; the older gfx110X-dgpu folder stopped at 7.10), which also ship AMD's real
`librocm_sysdeps_*`, so the libdrm/libnuma rebuild is gone. On the matched runtime: queued executions,
1,000-update compiled loops and interpreter exit all exit 0. Steady-state training (synthetic data,
`python -m checks.train_speed_check 512 1000`): ~460 updates/s at batch 512 vs PyTorch's 767.
Caveat: nightly wheels may disappear from AMD's index; the pinned hashes keep builds reproducible while
the files are in the store. Stable 7.13.0 exists on repo.amd.com/rocm/whl/gfx110X-all.

## 2026-11-25: JAX training at PyTorch speed

Steady-state, synthetic data shaped like the real thing (`python -m checks.train_speed_check 512 1000`),
batch 512: ~460 updates/s -> 639 after dropping the redundant goal-minus-observation input (the first
layer is linear, so it adds nothing; 9,240 -> 6,160 inputs) -> 955 with bfloat16 matrix multiplies
(float32 parameters and accumulation). PyTorch's eager fp32 reference was 767; at equal precision JAX is
~83% of it. Quality unchanged (Test A: held-out loss 2.438 vs 2.46 before; trips 58% vs 55-60%).
bfloat16 is for training only: on the CPU, where the player acts, it made a decision 415 us vs 153 us
in float32. Building training data by replay (~9,400 steps/s, one process) is now the slowest stage.

## 2026-11-25: more data alone doesn't make a better goal-reacher

Training data is now built straight into GPU memory (`experience.build(on_device=True)`; 60K recordings /
4.7M steps peak at 17.7 GB host vs being killed at a 30 GB cap before) at ~95K steps/s. Test A
(`python -m checks.policy_trips_check policy-data --train N --updates U`, ~160 trips each, +-4%):

| recordings | updates | held-out loss | trips reached (random) |
|---|---|---|---|
| 4K | 5K | 2.437 | 58% (45%) |
| 20K | 20K | 2.272 | 61% (44%) |
| 60K | 40K (batch 512) | 2.234 | 54% (41%) |
| 60K | 10K (batch 4096) | 2.295 | 57% (41%) |

Loss improves slowly with data; trip success stays ~55-60%, 13-16 points over random, far below
exact replay (92%). Hindsight from random play is capped by how poor a teacher random play is. Next:
self-improvement (GCSL-style): the player makes goal-directed trips, they're relabeled with where they
actually went, and it retrains on them.

## 2026-11-25: why copying random play caps the player

- **Self-improvement** (`python improve.py policy-data --rounds 4 --train 20000 --trips 20000`, 600 fixed
  held-out trips; random 42%, exact replay 92%): learned player 57% -> 61% -> 59% over rounds 0-2, then
  GPU out of memory in round 3 (previous round's 8.7 GB data still alive; fixed).
- **"Already matches the goal?" inputs** (one flag per byte; not linear, unlike the dropped difference
  input): round 0 rises from 57% to **65%** of trips (closing 73% -> 81% of the distance). Training 743
  updates/s (1.5x wider input).
- **Gap diagnostic** (`python -m checks.gap_accuracy_check policy-data`): predicting the pressed button is
  ~21% right at every goal distance 1..32, flat.
- **Ceiling** (`python -m checks.button_ceiling_check policy-data`): pressing each of the 25 button combos
  for one step from sample states gives only ~7.5 distinct outcomes (in the chosen bytes), so no model can
  name the pressed button more than ~30% of the time; ~33% after 4 steps. The model is at ~70% of that.
  No pipeline bug: most buttons are interchangeable at any moment, so "which button did random play
  press?" is mostly unanswerable, and imitating it yields a blurry, dithering player. Next: score actions
  by how likely they lead to the goal (contrastive goal-conditioned RL) instead of imitating random play.

## 2026-11-25: the missing piece is a distance, not search depth

Same 600 held-out trips (random 42%, exact replay 92%, imitation player 62%).
- **Contrastive critic** (`critic.py`, separate state-action and goal encoders, InfoNCE, geometric futures
  mean 16): 49%. Separate encoders can't compare state and goal byte by byte, and negatives from other
  recordings let it win by recognizing rooms.
- **Search player** (`trips._searched`; tries all 25 buttons held N steps in the emulator each step, keeps
  the best outcome; uses save states, so a teacher, not a way to play; `python -m checks.search_trips_check`):

  | distance used | hold 1 | hold 2 | hold 4 |
  |---|---|---|---|
  | memory bytes matching the goal (generic) | 58% | 58% | 60% |
  | hand-made place distance (diagnostic) | 54% | - | **83%** (closes 92%) |

  Bugs fixed on the way: the hold length set in the main process never reached worker processes (all
  runs used the default), and ties went to the first button ("press nothing"), so a coarse distance stood
  still (39% before random tie-breaking). With a good distance, greedy 16-frame lookahead nearly matches
  exact replay: the bottleneck is the distance measure. Next: learn a distance (steps apart) from
  recordings and search with it.
- **Learned distance** (`distance.py`: same inputs as the policy, classifies the step gap into buckets
  0/1/2-3/4-7/8-15/16-31/32-64/far; pairs 0-64 steps apart in a recording plus 25% from other recordings;
  20K recordings, 20K updates): search with it (hold 4) reaches **71%** of trips, closing 87% of the
  distance, vs 60% for matching bytes and 83% for the true distance. Best generic result so far.

## 2026-11-25: iteration speed (`python -m checks.stage_timing_check policy-data`)

| stage (4K recordings, 200 trips) | before | after |
|---|---|---|
| choose bytes | 6.0s | 3.3s (incl. starting the worker pool once) |
| sample 200 trips | 2.8s | 0.2s |
| build 4K recordings | 6.3s | 4.0s |
| train player / distance, 5K updates | 11.9 / 16.4s | 7.6 / 11.8s |
| 200 random / learned / search-learned trips | 2.4 / 12.5 / 13.2s | 0.1 / 0.9 / 4.5s |

Causes, each measured before fixing:
- Trip workers launched one after another (ready 0.4 s to 8.3 s): the player's 19 MB went as a startup
  argument, and each launch blocked until the previous worker had read it. Now via a /dev/shm file.
- More workers made learned trips slower (1 worker 2.1 s, 18 workers 9.0 s): one decision at a time
  reads the whole 19 MB first layer, so 18 workers saturate memory bandwidth (~34 GB/s). A 2 MB player
  scaled fine. Workers now run up to 32 trips in lockstep and decide for all in one batch.
- Every pool start re-imported the main script, and so JAX, in 18 processes (2 s vs 0.09 s without JAX),
  and every build/sample/run started a pool. Now one pool per run (`workers.py`).
- The workers' CPU-only JAX settings never applied (JAX was already imported via the main script), so all
  workers initialized the GPU. Now set in the environment workers inherit.
- Compiling training code: JAX's disk cache (flake env) saves ~4 s per model on later runs.
Unchanged behavior: random and replay players give exactly the same 42% / 92% on the 600 held-out trips.

## 2026-11-25: first distillation of search (`python distill.py policy-data --distance-train 20000 --distance-updates 20000 --trips 5000 --policy-train 20000 --policy-updates 10000`, 4m20s)

Search teacher with the learned distance: 73% of the 600 held-out trips. Players with no lookahead,
trained with hindsight goals:
- random play only (20K recordings): 60%
- random play + 5K teacher trips: 59%
- 5K teacher trips only: **64%**

A small gain from far less data (5K short trips vs 20K recordings), but the labels are still blurred:
the recorded buttons include the 25% sticky-noise repeats, and hindsight goals ignore the goal the
teacher was actually aiming for. Next: record the teacher's own choices and train on its real goal.

## 2026-11-25: what students should learn from the search teacher (`python distill.py policy-data`)

Teacher (learned distance + 4-step lookahead): 73% of the 600 held-out trips. Students, no lookahead:

| student | 5K teacher trips | 10K teacher trips |
|---|---|---|
| hindsight goals, buttons as pressed | 64% | **68%** |
| teacher's real goals, teacher's chosen button | 56% | - |
| teacher's real goals, regress all 25 button scores | - | 59% |
| same, scores minus the step's average | - | 57% |

(random-play-trained player: 60-62%.) Real-goal students memorize: 33% of the teacher's buttons right on
trained-on lessons vs 14% held out, and held-out loss worsens with training (2.84 -> 3.02). Real goals give
one goal per trip; hindsight gives many per trip. Teacher choices are also spread (no button over 9%).
More teacher trips help hindsight students: 64% -> 68% for 5K -> 10K.

## 2026-11-25: what students should learn from the search teacher (`python distill.py policy-data`)

Teacher (learned distance + 4-step lookahead): 73% of the 600 held-out trips. Students, no lookahead:

| student | 5K teacher trips | 10K teacher trips |
|---|---|---|
| hindsight goals, buttons as pressed | 64% | **68%** |
| teacher's real goals, teacher's chosen button | 56% | - |
| teacher's real goals, regress all 25 button scores | - | 59% |
| same, scores minus the step's average | - | 57% |

(random-play-trained player: 60-62%.) Real-goal students memorize: 33% of the teacher's buttons right on
trained-on lessons vs 14% held out, and held-out loss worsens with training (2.84 -> 3.02). Real goals give
one goal per trip; hindsight gives many per trip. Teacher choices are also spread (no button over 9%).
More teacher trips help hindsight students: 64% -> 68% for 5K -> 10K.

Seed noise check (hindsight students on the cached 20K-trip teacher, 600 held-out trips):

| teacher trips | 10K updates, seeds 1/2/3 | 30K updates |
|---|---|---|
| first 10K | 61 / 61 / 63% | 65% |
| all 20K | 63 / 66 / 64% | 66% |

The earlier 68% (10K trips) was a lucky draw; seeds and data move results ~3 points. Distilled students
sit at ~63-66%, a few points over random-play-trained players (60-65%) and well under the teacher (73%).
Short-trip success is not where big gains are; next is chaining trips from power-on (Test B).

## 2026-11-25: Test B, start of game to Morph Ball, no save states (`chase.py`, 100 attempts, sticky 0.25)

| setup | reached Morph Ball |
|---|---|
| archive chain as waypoints (4), any player | 0% (best learned attempt: 3/4 waypoints) |
| clean stitched route (`route.py`, 132 steps), waypoints every 10 steps, distilled student | 7% |
| same route shortened to 101 steps (coarse-to-fine waypoints) | **25%** (spacing 20: 20%) |
| + one round of practice around the route (`practice.py`) | **70%** (rounds 2-3: 62%, 67%) |

- Per-hop test on the archive chain: the learned player handles 3 of 4 hops (76-82%) but the 2nd hop's
  waypoint (an odd mid-jump spot exploration happened to save) is 0-5% for every player, even search with
  the true distance. The route, not the player, was the main bottleneck.
- Route stitching: grow a replayable path hop by hop from the best (furthest, soonest) of 200 attempts.
  Shortening re-stitches with the path's own states as waypoints, sparse first (dense waypoints make the
  player retrace detours): 132 -> 101 steps, about 6.7 s of game time. Still some dawdling.
- Practice: from 1000 random route points knocked off by 0-15 random buttons, keep the fastest of 8
  attempts to rejoin 1-3 waypoints ahead (95-100% recover); retrain (hindsight) on teacher trips + these.

## 2026-11-25: shorter route; advantage weighting

- Shortening with the practiced player, 1000 attempts per round, spacings 100..20: 101 -> **67 steps**
  (~4.5 s of game time; first 132). `python route.py policy-data --resume --player ... --attempts 1000`.
- Practice on the 67-step route (1000 knocked-off starts, 993 recovered, 450K emulator steps), then two
  players on identical data (`python practice.py policy-data --compare --player ...`), Test B, 100 attempts:
  plain hindsight **63%**, advantage-weighted (learned distance, AWR, temperature 1) **51%**. No gain; the
  distance (trained on random play) is probably too noisy per step to grade single moves.

## 2026-11-25 (night): design review; a no-learning route follower wins

Independent review (second agent) points: the learned parts only follow routes the random archive found
(system ceiling = random Go-Explore's ceiling); hand-made places are load-bearing in training (breaks the
"works on Zelda" test); a third of the code is logged failures.

- **Generic "reached" test** (`python -m checks.reached_check policy-data`): agreement with hand-made place
  equality on held-out pairs: learned distance AUC 0.75, share of matching bytes AUC 0.84. Neither is a
  drop-in replacement.
- **Automatic place bytes** (`python -m checks.slow_bytes_check policy-data`): ranking bytes by slowness, or
  by entropy / change rate, does not find map x/y (ranks ~1500-2650 of 3093): many bytes differ between
  archive states yet rarely change within a recording. Per-byte statistics: ruled out.
- **Route follower, no learning** (`route.Follower`: press the route's button at the route state with the
  most bytes equal to the current memory; `python chase.py policy-data --route ... --follower --players
  learned-greedy`): **92%** to Morph Ball from the start, vs 70% for the practiced student and **35% for
  blind replay** of the same buttons under the same sticky noise. Closed-loop nearest-neighbour lookup on raw
  RAM beats the trained network by a wide margin (cf. episodic control).

## 2026-11-25 (night): archive keeps paths; complexity review and pruning

- **Paths in the archive** (Go-Explore's original design): every place keeps the shortest replayable button
  path from the start state; workers rank candidates by path length. 1-minute run: 200/200 sampled paths
  replay to their place (`python -m checks.path_check <run>`), median 522 steps; speed unchanged (~490K
  frames/s). This replaces stitching routes from archive parent chains, which got stuck after 14-18 of 110
  waypoints toward missiles (random or learned attempts) on odd waypoints (door transitions, near-death
  lineages).
- **Random attempts can stitch** the Morph Ball route (367 steps) but cannot shorten it; the learned student
  stitched 132 and shortened to 67. Learning helps route *finding*, not route *following*.
- **Complexity review** (Hickey / Ousterhout / Grug): `trips.py` braids the trip runner, a 7-way player
  string switch and Metroid-only success tests; exploration has its own actor/pipe workers beside
  `workers.py`; "route / path / chain / waypoints" name one idea (start state + buttons); `chase.py` grew 6
  modes. Removed now: keep rules (empowerment, rare-empowerment, safety, more-missiles, more-health) and
  scores, selection modes (productive, powerful), `empowerment.py`, `vitals.py`, `promise.py`, `critic.py`,
  `improve.py` and their checks (all logged failures; in git history). Deferred until the path-follower
  results settle what survives: players as values with one interface, injected success test, one worker
  system, one path concept.

## 2026-11-25 (night): following archive paths; compounding

`runs/paths-15min` (shortest-path archive, 15 min): 87 rooms, Morph Ball, 10 max missiles, Norfair;
300/300 sampled paths replay. Shortest stored paths: Morph Ball 58 steps, missiles 2192 steps.

| follower on the archive path (100 attempts, sticky 0.25) | reached |
|---|---|
| Morph Ball (6 waypoints) | 60% (vs 92% on the 67-step stitched route) |
| missiles (220 waypoints) | 0%; median 43/220 waypoints, best 93; 20% died |
| same, follower searching only the 30 states before its waypoint | 0%; median 27/220; 4% died |

Per hop (30 attempts each, from the path's own state): most hops 87-100%, only 2 of 220 under 50%, yet
the product over all hops is ~0. Failure is spread thin: a single route followed closely compounds to zero
over long horizons (99%/hop over 220 hops = 11%; the game is ~50x longer). Shortening the 2192-step path
with the student got stuck (it never learned that territory). A player that can recover from anywhere
near the route is required, not a better route.

## 2026-11-25 (night): the exploration wall is not resources or budget

- **Resource-aware places** (hand-given diagnostic: current missiles + energy bands; `runs/resources-15min`,
  15 min): 49,448 places (6x), 77 rooms, Morph Ball, 10 max missiles; nothing new. Wall unchanged.
- **Budget?** (`python -m checks.armed_check resources-15min 3000 10`): 30,000 random segments from 3,000
  states holding 5+ missiles: **0** reached a new room, item, area or missile capacity. Not a selection
  budget problem; the explored region is saturated for random play.
- **Frontier** (`python -m checks.frontier_check`): 14 room edges face unvisited map cells; nearly every
  state there has 2-3 energy and 10 of 14 never held a missile. New territory is only reached by
  lineages that spent everything.
- **Transport** (`python -m checks.transport_check`): carrying the strongest state on a door's stored path
  to the door with the follower (best of 50 per hop, save states allowed) worked for 3 doors (5-9
  missiles, 23-36 energy at the edge): **0 of 600** random segments from there reached anything new.
  A scripted probe (switch to missiles, shoot, walk through) at one such edge didn't pass either: unclear
  whether these edges are real passages; asked a research agent for the game's layout facts.
- **Follower with remembered recoveries** (`chase.py ... --recovery-rounds 2`): 48K entries, every attempt
  stuck at waypoint 8/220 (vs median 43 without): a recovery entry near the route poisons a deterministic
  lookup. Parked: playback only matters once exploration can reach further.

## 2026-11-25 (night): the wall is distance in place x inventory, not waste

- **Long Beam red door** (map y 5, x 7; facts from the disassembly via a research agent: 5 missile hits,
  beams don't count, opens for good): room (5,8) beside it is reached 80+ times in every run, **never with
  a missile**; room (5,7) never entered.
- **Causal test** (`python -m checks.red_door_check resources-15min 300`): 20 door states x 300 random
  segments: 0% enter (5,7) as found; **0.18%** (11/6000) with 5 missiles edited in. Random play can open it
  when armed.
- **Frugal exploration** (diagnostic: SELECT combos 10x rarer, + resource places; `runs/frugal-15min`):
  80 rooms, 60,948 places, 48 states at the door, all with 0 missiles. Wasting missiles is not why.
- Conclusion: missiles come from far away (room 11,18); carrying them back means re-crossing territory that
  is only "new" because of the inventory. Exploring place x inventory with uniform random segments is
  multiplicative in items: hopeless for the whole game. Needed: composition, i.e. travel across known
  territory with what you carry. Second-agent review agreed and added: playback needs re-planning, and timed
  skills (bomb jumps, ice platforms) will need reusable action chunks.

## 2026-11-25 (night): the archive as a map (`graph.py`): travel is brittle

Locations (place minus inventory) and edges (button chunks between them) from replaying 3000 stored paths
+ 20,000 recorded segments: 3,397 locations, 17,645 edges (stored paths alone only lead outward from the
start, so there was no way back). `graph.travel` plans the shortest chain and makes each hop by known
chunks, then random retries, accepting arrival further along, banning failed hops and re-planning.
`python -m checks.travel_check resources-15min 18 300 3000`: 0 of 18 armed states reached the Long Beam
door. First version: hops taken by the first chunk that works spent everything (arrive 0 missiles, 0-3
energy). Keeping the arrival that holds the most (hand-given missiles, energy) preserved missiles but
journeys still drained energy and bounced inside Norfair rooms until the re-planning limit. Fixing hops
one by one is hill-climbing on map details; parked in favor of restarting exploration per acquisition.

## 2026-11-25 (night): exploration layers from the first missiles (`--start-state`)

Fresh exploration from a state just after the first missile pickup (room 11,18; 5 missiles, 36 energy;
`cache/start-first-missiles.state`), 15 min each:
- resource places (missiles + energy bands): 52 rooms, 54,648 places; never reached the door's row.
- plain places: **88 rooms** (most of the night), 5,156 places; door room (5,8) reached 44 times, all with
  0 missiles. Random play spends the 5 missiles on the long walk back, even when every lineage starts
  armed.
- plain places + frugal play (SELECT 10x rarer): **94 rooms** (most of the night), 7,532 places; door room
  reached 86 times, all with 0 missiles. The seed starts with missile mode off (B alone fires nothing), but
  a rare SELECT leaves missile mode on for hundreds of steps, so the walk back drains them anyway.
- Conclusion for the night: every random-exploration variant (resource places, frugal, fresh layers, map
  travel) delivers the Long Beam door empty-handed. Carrying a resource across known territory to where it
  is needed requires deliberate behavior; random play with save states does not provide it.

## 2026-11-25 (morning): hoarding carries missiles to the door

Josh's framing: hoard anything spendable, without proof it's useful (cf. RPG potions, a mathematician's
lemmas). Implemented as an archive keep rule, not behavior: `--keep-more <memory index>` ranks a place's
candidate states by the value of that byte (more is better), then path length. Exploration still presses
random buttons (spends freely); the archive just remembers the richest visit to each place. Diagnostic:
the missile byte is named by hand (index 4217); the generic version would use automatically detected
spendables.
- Layer from the first missiles, plain places, hoarding missiles (`runs/layer-keep-15min`): 94 rooms;
  **4,001 of 4,468 places hold 5 missiles; the door room has 32 states holding 5** (0 in every earlier
  variant). No entry beyond the door in 15 min.
- Opening depends on height (`python -m checks.red_door_check resources-15min 300`, per state): all 11
  successes came from 3 states high in the room (room y 52-87), ~1-2% each; 17 lower states: 0/300.
- **Long Beam!** Same setup run for 45 min (`runs/layer-keep-45min`): the Long Beam at **176 s**; then a
  stall: 97 rooms, 16,875 places, 10 max missiles, no Bombs in the remaining 42 min. Hoarding only
  missiles let energy run down everywhere (the best Long Beam state with 5+ missiles had 14 energy).
- `--keep-more` now takes several "index:scale" terms, summed (hoard missiles and energy together).
- **Finding spendables automatically** (`python -m checks.consumables_check policy-data`, 3000
  recordings): "decreases far more often than increases, in small steps" puts missiles 27th and energy
  (tens) 24th of ~10K bytes; also requiring persistence across scene changes doesn't help (the junk persists
  too); requiring rare changes (< 0.2% of steps) lifts them to **10th and 11th**. A paired test (two
  random plays from one state end different) separates missiles (4%) from most junk (0-1%) but not energy
  (1%). Next: hoard the automatic top 12, each divided by its largest observed value, and see whether the
  Long Beam still falls.

## 2026-11-25 (midday): layer 2 gets Bombs; generic hoarding needs precision

- **Layer 2** from the strongest Long Beam state with 5+ missiles (7 missiles, 14 energy;
  `cache/start-long-beam.state`), hoarding missiles + energy tens (`--keep-more 4217,263`), 30 min
  (`runs/layer2-keep-30min`): **193 rooms, Brinstar + Norfair + Kraid, Bombs, Long Beam, 1 energy tank,
  15 max missiles**. Second red door opened. Twice the rooms of any earlier run.
- **Generic hoarding, automatic top 12** (two seeds, 15 min from the first missiles): 114 / 119 rooms, an
  energy tank, **no Long Beam**. The door room's states hold 0-2 missiles (hand hoarding: 5-10): missiles
  are 1 of 12 summed terms, so the junk bytes decide what is kept. Hand hoarding of missiles alone, in turn,
  left median energy at 1-2; generic hoarding kept it at 13-21.
- Detector precision: requiring that two random 120-step plays from one state end with the byte different
  in at least 2% of states lifts **energy (tens) to 2nd and missiles to 3rd** (behind bytes 35 and 40).
  Running: generic hoarding with the automatic top 4 (35, 40, energy tens, missiles), two seeds.
- **Automatic top 4 (sum)** (`runs/hoard-top4-s1`, 15 min): 119 rooms, energy tank, median energy 64, but
  door states hold 0/2/4 missiles, never 5: the sum traded a missile for energy. A red door needs exactly 5.
- **Pareto hoarding**: each place keeps every visit no other visit beats on all hoarded bytes (equal
  holdings: shorter path wins), so nothing is traded. First version appended a new entry whenever a visit
  beat a kept one: ~50K stored, ~4.5K active over ~2K places after 5 min (3 seeds x 6 workers, `race.py`);
  now a beating visit takes over the beaten one's slot. `--stop-at-item` ends a run at a milestone.
- Fast-protocol lesson: 3 seeds x 6 workers x 5 min equals ~1.7 min of a full run: too short for the
  Long Beam (3 full-run minutes before); judge by proxies (armed visits at the door), not the rare event.
- **Pareto with place-first selection** (3 seeds x 6 workers x 8 min, vs missiles-only hoarding): Pareto
  median energy 24-33 but door visits hold **0** missiles; missiles-only: median energy 2-4, door visits
  mostly 3-4 missiles. Pareto spreads effort over healthy and armed lineages, and the healthy ones that spent
  their missiles reach the door first. A trade-off in effort, not a bug. Also: 8 min x 6 workers (~2.7
  full-run minutes) is too little budget for long-distance effects like carrying missiles back.
- **Rarity order**: measured increase rates, rarest first: energy tens 2.1e-5, byte 35 2.6e-5, byte 40
  3.4e-5, missiles 4.3e-5 per step (enemies drop missiles). "Prefer the rarest" would put energy first:
  nothing generic says missiles matter more; only the door does.
- Testing instead: **one exploration per detected spendable** (portfolio: each hoards only its own byte),
  4 x 4 workers x 25 min from the first missiles, stopping at the Long Beam.
- **Portfolio works** (`runs/portfolio/`): from the first missiles, one exploration per automatically
  detected spendable (top 4), 4 workers each, 25 min, stop at the Long Beam:
  byte 35: 103 rooms, energy tank; byte 40: 76 rooms; energy tens: 83 rooms; **missiles: Long Beam at
  941 s** (85 rooms). No resource named by hand: hoarding each candidate separately, and letting whichever
  branch opens something new win, avoids having to know which resource matters. Still hand-given: the layer
  start state, the "top 4" cutoff, and hand-made places. No NES Zelda ROM on unit (Josh to provide) so the
  Zelda rule can't be measured yet.
- **Automatic layer trigger, not solved yet** (`python -m checks.one_way_check <runs joined by +>`):
  bits that only ever go 0 -> 1 in recordings include the Morph Ball and missile-capacity bits but also
  300-500 junk bits (never seen flipping back in short recordings); the Long Beam and Bombs bits never
  appear, since sampled recordings rarely include those pickups. "Most one-way bits on" picks Bombs visits
  in layer 2 but not Long Beam visits in layer 1; "rarity-weighted one-way bits gained since the layer
  start" picks Long Beam visits in layer 1 but not Bombs visits in layer 2. Needs a better signal, e.g.
  watching every step for bits that flip on and stay on, instead of sampled recordings.

## 2026-11-25 (afternoon): Bombs reproduced; acquisitions are clusters of one-way bits

- Every-step bit watching (`progress.see_change`, bitwise on bytes, ~8% slower; saved as bit_flips.npz)
  leaves 210-300 one-way bits after a 15-20 min run, vs ~1000 after 1 min, and catches item bits that
  sampled recordings never saw.
- Layer 1 rerun (missiles-only hoarding, 15 min, `runs/layer1-watch`): 95 rooms, no Long Beam (Long Beam
  now 2 of 4 such runs). Layer 2 rerun from the Long Beam (missiles-only hoarding, 20 min,
  `runs/layer2-watch`): **Bombs again** (129 rooms, energy tank) - reproduced.
- Trigger: summing or maximizing rarity over gained one-way bits picks junk held by one or two visits.
  Real acquisitions *spread*: in layer 2 the Bombs bit (0x1078.0) is held by 14% of visits **together
  with a block at 0x1086-0x1094** (cartridge RAM where the game records taken items and opened doors),
  all exactly 14%; the energy tank bit 0x1077.0 with 0x108f-0x1090, all 55%. In layer 1 the second
  missile tank: 0x107a.1/.3 with 0x108b-0x108c, all 53% (plus noisier map/room bits). Next: define an
  event as a set of one-way bits held by exactly the same visits; each event seeds a child layer.
- **Events** (`events.py`: one-way bits the layer start lacks, grouped by exactly which visits hold them):
  layer 2 -> 10 events; **Bombs** = 6 bits (items byte + 2 bytes of the game's taken-items record, 14%),
  **energy tank** = 6 bits (tanks + record, 55%), and an 8-bit event at 14% whose first holder lacks Bombs
  (likely the red door before Bombs recorded as opened). Layer 1 -> 19 events; the second missile tank is
  the 3rd largest (capacity + record, 53%); the two larger ones are map/room-state bits (one-way only within
  the run). Plan: every event seeds a child layer from its best holder; noise events become branches that
  stall.

## 2026-11-25 (evening): the automatic climb

`climb.py`: layer 0 from power-on; spendables detected on its recordings; each layer runs one hoarding
branch per spendable side by side; each new event (events.py) seeds a child layer.
- Detector fixes, all checked offline on cached recordings: rank by control judged only where the byte is
  held (missiles 52%, energy tens 29%, junk 8-10%) instead of decrease/increase ratio; 6000 shuffled control
  starts (2000 in stored order picked junk; missiles held in only ~2% of states). Missiles now first on
  3/3 shuffles. Energy tens drops out (changes slightly too often for the "rare" gate).
- Events: cap share at 90%: from power-on the largest events are start-area bits held by 93-96% of visits;
  the two missile tanks are the real ones (22%, 29%).
- `runs/climb3` (reusing climb2/layer0): layers from both missile-tank events, 4 branches x 4 workers x 12
  min: no Long Beam (missile branch best each time: 86 and 71 rooms). Budget too small: the earlier
  portfolio's missile branch needed ~16 min at 4 workers; 75% of compute went to junk branches.
  Next: top 2 spendables, 8 workers each, 15-min layers.
- `runs/climb4` (top 2 spendables x 8 workers x 15 min, stall stop 3 min): layer 1 (second missile tank, in
  Norfair): missile branch stalled at 332 s with 28 rooms while the other branch reached 90; layer 2 (first
  tank): 83-87 rooms, no Long Beam in 12 min. Stopped. The stall rule may be too aggressive for slow starts.
- **Playback check** (`chase.py ... --route cache/journey-to-bombs.pkl --target bombs --follower`): the
  route follower on the stored 18,821-step power-on-to-Bombs path, 50 attempts: median 17 of 1,883
  waypoints (best 23, ~200 steps). The save-scummer's path is reachable but not playable as is: it needs
  shortening and re-planning (phase 2).

## 2026-11-25 (evening): surprise didn't help; "choose like a hoarder" moves the real bottleneck

- **Surprise** (`--surprise`: selection bonus for visits whose segments show a rarely-changing byte taking
  a never-seen value; whole change patterns were far too noisy: 102K new patterns in 90 s): race from the
  first missiles, 3 seeds x 6 workers x 15 min: base Long Beam 1/3 (at 619 s), surprise 0/3, 17% slower.
  Door-room proxy over the 6 runs: 4 reached the door with **0 missiles**; the bottleneck is arriving armed,
  not hitting the door, and surprise can't help before anything reacts.
- **Fast harness** (~4 min): start 400 steps before an armed door arrival from a successful run
  (`cache/start-near-door.state`: room 7,11, 4 missiles, 1 energy), many short runs in parallel, score the
  most missiles held in the door room.
- **Choose like a hoarder** (`--favor-holdings W`: selection weight x (1 + W x holdings / max holdings)),
  6 seeds x 2 workers x 3 min each side: base reached the door 6/6 but kept all 4 missiles in 1/6; favor
  (W=10) reached it 4/6 and kept all 4 missiles in **3/6**. Arrives armed far more often, spreads a bit slower.
- Armed near-door start (`cache/start-near-door-armed.state`: room 7,12, 10 missiles, 2 energy), 6 seeds x
  2 workers x 3 min each side: Long Beam 0/6 base, 0/6 favor. Door visits / armed / armed high up per seed:
  base 0/0/0, 32/18/0, 31/0/0, 42/0/0, 44/0/0, 32/32/0; favor 6/0/0, 0/0/0, 25/23/0, 0/0/0, 0/0/0, 32/16/0.
  **No armed visit ever got high in the door room**, and the door only opens from up high (all 11 earlier
  successes came from room y 52-87). The chain is: arrive armed (favor helps) -> climb high while armed
  (now the failing link) -> 5 hits (~1-2% per try).
- **Carrying to the door** (`python -m checks.carry_check cache/start-near-door-armed.state "fast3/*"`):
  map from the 12 short runs (1,801 locations, 12 high spots in the door room); graph.travel from the armed
  start keeping the arrival with the most missiles (then energy): **kept all 10 missiles over 14 hops**, but
  stuck in room (8,11) at the same hop both times (409K steps of retries): a hop that needs a precise climb,
  which neither recorded chunks nor 200 random 60-step tries make from that state. Resources survive travel
  now; skillful hops don't.

## 2026-11-25 (night): carrying an armed state to the door makes the Long Beam reliable

- **Local exploration for hard hops** (`graph._local_search`): when no recorded chunk or quick random try
  makes a hop, a small Go-Explore from the current state (locations, richest visit kept, sticky random
  segments) returns the furthest-along and richest of its first 20 arrivals. First version returned the
  first arrival: reached the door room after 38 hops but spent all missiles.
- `python -m checks.carry_check cache/start-near-door-armed.state "fast3/*"`: the armed start (10 missiles,
  room 7,12) was **delivered high in the door room (room y 87) holding 5 missiles**, after 37 hops
  (529K emulator steps of hop attempts, ~6 min, one process). 300 isolated random segments from there: 0
  openings (2 energy: short tries die).
- **Exploration from the delivered state** (`cache/carried-to-door.state`, missile hoarding, 3 seeds x 6
  workers): **Long Beam in 3/3, at 42, 70 and 124 s**. Versus 1/3 in 15 min from the missile pickup without
  carrying. Composition (carry what you hold to where it's needed, then explore) turns the chain of luck
  into a deliberate step. Still diagnostic: the target (high spots in the door room) was picked by hand.
- **Choosing carry targets without hand-picking**: "barriers" = dead-end locations (no visit there has
  descendants) only ever reached poorer than the richest state, ranked by how often exploration tried them.
  Short runs (3 min x 2 workers): 255-329 such places, tries flat (<= ~20), no signal. Long run
  (`layer-keep-45min`): 101 such places; **the door room ranks #2** (950 tries); top 5 have ~900-1000 tries
  each. Plan: after each layer, carry the richest state to the top few barriers in parallel, then explore from
  each delivered state.
- **Barrier carry, end to end** (`python -m checks.barrier_carry_check <run> 5 2`, `barriers.py`):
  on `layer-plain-15min` (no hoarding): top barriers were Norfair spots near the layer start, and carrying
  spent everything (hop arrivals were ranked furthest-first). Fixed to rank arrivals riches-first. On
  `layer-keep-15min` (missile hoarding): delivered 10 / 6 / 9 / 9 / 3 missiles to the top 5 barriers, but
  those were all low spots in Norfair rooms next to lava: "tried often, nothing beyond" also describes
  **death traps**. Barriers must be where exploration survives yet finds nothing new; needs per-visit death
  counts (not recorded yet).
- **Death traps vs barriers** (seconds, on saved states; `barriers.find` estimates deaths with probes when
  an archive lacks counts): filtering out spots where >50% of 20 random 100-step probes die removed the door
  too: door visits kept in `layer-keep-45min` have median energy 4 and die 40% of the time; with energy
  edited to 39, 0%. A death filter judges the visitor, not the place. Better: carry first (the carried state
  arrives with resources), and let real death traps fail as cheap branches.
- Ranking poor dead ends by tries (wealth = missiles, no death filter): door room **#2 of 60 in the 45-min
  layer, absent from the top 60 in a 15-min layer**: the barrier signal needs long layers (enough armed
  tries at the door).
- Barrier rankings (seconds, offline), door room's best rank among poor dead ends:
  | ranking | 15-min layer (2,472 dead ends) | 45-min layer (101) |
  |---|---|---|
  | times tried | 131 | **2** |
  | shortfall (richest - best held) x tries | **30** | 37 |
  | shortfall, then tries | 75 | 69 |
  No single ranking puts the door near the top in both; a portfolio would need the top 30-40. Paused here:
  open question is a sharper general "where are resources needed?" signal.
- **Carrying inside exploration, sampled targets** (offline): the door room gets only 1.2-1.9% of the
  sampling weight (by tries or shortfall x tries) -> ~120-200 carries for a 90% chance of one door target;
  no better than hoarding + luck. Not built.
- **"Watch the barrier as it fails"** (`python -m checks.reaction_check layer-keep-45min`, 7 s): reactions =
  steps where a rarely-changing byte takes a value unseen earlier in the data, counted per location: 1,331
  reactions at 287 locations; **door room ranks 115**. First visits to any room produce new values everywhere,
  and door hits are rare in sampled recordings. Not a sharp signal either.
  Open problem: a general signal for *where a resource is needed*.
- **Need = the counterfactual "does holding more let me get somewhere new here?"** (`python -m
  checks.need_check layer-keep-15min 4217 60 300`, **24 s**): for the top 60 poor dead ends (shortfall x
  tries), 300 random probes from a visit as found vs the same visit with the detected spendable edited up to
  the richest amount; need = extra probes reaching locations the archive never saw. **The door room ranks
  #1, #2 and #3**, the only candidates with need > 0 (3, 1, 1). Sharp where tries, shortfall, death rates and
  reactions all failed. General: it edits only the automatically detected byte and asks where having more
  of it opens new territory (in Zelda: bombable walls, locked doors).
- **End to end with no hand-picked target** (`python -m checks.barrier_carry_check layer-keep-15min 3 2
  --need`, 4.5 min): need ranking -> top 3 are the door spots; carrying from the *nearest richer* state
  (fewest planned hops among visits holding more than the target ever had; carrying from the far richest
  state failed after 8 hops) delivered 5/4/4 missiles to all three; **exploration from each got the Long
  Beam in 64, 2 and 38 s**. On an archive that never got the Long Beam by itself in 15 min.
  Loop: explore with hoarding -> need-rank dead ends (counterfactual probes) -> carry from the nearest richer
  state -> explore.
- **Second-agent review (evening)**: the design is becoming a pile (~10 mechanisms, most are Go-Explore with
  a different keep rule, pick rule or start). Unifying frame: **"cheat, then pay back"**: edit what you hold
  (spendables to max, event bits on, maybe position) in saved states; cheats that open new places become
  debts ("be at L holding >= k of R"); one goal-directed Go-Explore (hard floor on R, pull toward L) pays
  them; layers = a debt got paid (relaxed planning, applied to RAM). Biggest hidden risk for Zelda: `location`
  still comes from hand-made places. Suggested: infinite-resources ceiling test; carry as a floored
  Go-Explore (replacing graph.travel / local search); teleport and event-bit "need"; mined button chunks
  for skills.
- Ceiling test, too small (`--refill 4217:5,262:0x99,263:0x03`, a diagnostic: missiles and energy rewritten
  at every segment start; 2 seeds x 4 workers x 5 min each): base 28/32 rooms, refill 70/34 rooms, no Long
  Beam either way. Rerunning at 9 workers x 10 min from the first missiles and from the Long Beam.
- **Ceiling with free resources** (9 workers x 10 min, missiles + energy rewritten at every segment start):
  from the first missiles: **Long Beam at 154 s** (96 rooms); from the Long Beam: **Bombs at 416 s** (120
  rooms, energy tank). Without the cheat: about a coin flip per 15 min, and ~20 min. Resources are the
  binding constraint at these stages, not skill or places: carrying (paying back the cheat) is the right
  lever, and a free-resource exploration is a fast way to find what opens next and where.
- **Second door (Bombs)** via need -> carry: a 5-min hoarding layer from the Long Beam (81 rooms) never
  reached the Bombs door room (0, 5, 25) (found as the room whose visits led straight to Bombs in
  `layer2-watch`), so all top-60 candidates have need 0 and carrying has no real target. Bug fixed on the
  way: maps must replay stored paths from the run's start file (Pareto slot reuse can replace the archive's
  first slot). The wall before Bombs is reaching the room (the lava corridors drain energy); the ceiling run
  with energy refilled got Bombs in 416 s. Next: the same need/carry loop with energy as the resource, i.e.
  carry health toward where exploration keeps dying.
- **Energy need** (`need_check lb-5min 263 ... [--from-poorest]`): byte-level energy tens has no candidates
  (every dead end was once visited near full health); testing from the poorest visit at each dead end, need
  is 0 everywhere. Limit of the need probe: 100 random steps only reveal short-range needs (a door next to
  you); health pays off over long trips (surviving the lava corridors to the Bombs room). Long-range needs
  call for a different counterfactual (e.g. the free-resource ceiling run itself, "cheat then pay back").

## 2026-11-25 (late): can "careful play" be learned cheaply from recordings?

`python -m checks.careful_check layer-keep-45min+layer2-watch 6000` (~300K steps; MLP on RAM, with or without
the button pressed; held-out AUC; energy/missile bytes named by hand as labels):
| horizon | label (rate) | state only | state + button |
|---|---|---|---|
| 10 steps | damage (10.7%) | 0.912 | 0.909 |
| 10 steps | missile spending (11.0%) | 0.807 | 0.832 |
| 3 steps | damage (4.4%) | 0.935 | 0.935 |
| 3 steps | missile spending (3.8%) | 0.832 | 0.864 |
Damage is very predictable from the situation and not at all from the single button: under random play,
where you are decides whether you get hurt. Per-button steering can't make random play careful; avoiding
damage takes multi-step behavior (a value function under the agent's own policy, i.e. real RL), or using
the state-level danger prediction to choose which states to keep and explore from.
- **Danger-guided selection** (`--danger-model cache/danger-model.pkl --avoid-danger 3`; model = memory-only
  damage predictor, AUC 0.91), from the Long Beam, 3 seeds x 4 workers x 6 min each side: base 39/29/44 rooms,
  safe 31/16/16 rooms; neither reached the Bombs door room. Avoiding danger also avoids the frontier. "Do the
  right thing" beats "avoid the wrong thing" (Josh).
- **Learned state value from the family tree** (`python -m checks.value_check layer-keep-45min 4217 0 5 0,5,8
  [--per-try]`, 15-40 s): MLP from memory to log(1 + descendants): held-out correlation 0.10; counterfactual
  missiles 0 -> 5: +0.001 (higher for 99% of visits, but negligible). Target confounded by age (early visits
  have more descendants). Discoveries per try (children / times chosen): correlation 0.08, no missile effect.
  Per-visit discovery is too sparse and noisy for 17K examples; a learned "is this state good" needs a denser
  signal (e.g. per-step rewards for new places along recorded segments, learned with bootstrapping) or more
  data.

## 2026-11-25 (late): hand-mixed drives over-concentrate; drives as senses for a learned chooser

- **Drive combinations** (`tools/drives.sh`: from the Long Beam, missile hoarding, 6 min x 4 workers, 2 seeds;
  power = favor holdings 10, safety = avoid danger 3, surprise 5, mastery 5 = 4p(1-p) bonus for visits whose
  segments sometimes find new places): none 43/36 rooms; all 14/16; leave-one-out 10-19. No run reached the
  Bombs door room. Multiplying several "focus here" weights over-concentrates exploration; no single drive is
  to blame. The budget and the rooms proxy are also too weak to separate drives (Josh's question; agreed).
- **Value model, training length** (`checks/value_check.py`): descendants target: training correlation 0.25 ->
  0.41 -> 0.76 at 1K/4K/20K steps while held out 0.14 -> 0.11 -> 0.08 (memorizing noise); per-try target:
  0.05/0.16/0.24 vs 0.02/0.08/0.14 (slowly improving, low ceiling). Per-visit targets are too sparse.
- **Drives as senses** (Josh: a learned system taking drives as inputs): logistic regression from the start
  visit's drive signals to "did the segment find a new place", 266K segments from the drive runs: held-out AUC
  0.868, but **leaky** (features come from the final archive, including the predicted segments' own outcomes;
  success rate carried most weight). Next: log drive signals at selection time, then learn the chooser online.
- **Learned chooser** (`--learned-chooser W`: drive signals logged at selection time, logistic model refit every
  2000 decisions on the last 50K, selection x exp(W x standardized logit)), from the Long Beam, 2 seeds x 4
  workers x 6 min: W=0 40/35 rooms, found-rate 0.87%/0.63%; **W=1 44/36 rooms, 1.07%/0.77% (~+23%)**; W=3
  35/19, 0.69%/0.48% (over-concentrates). Clean offline check (base runs' decisions.npz, 220K decisions):
  **held-out AUC 0.949**; weights: rarely chosen +0.86, success rate +0.12, death rate -0.10, path length
  -0.06, holdings/danger/surprise ~0. The learner rediscovered Go-Explore's least-chosen rule; power doesn't
  predict *immediate* discovery because resources pay off far away. Next: train the chooser on downstream
  outcomes (did the lineage later get an acquisition / open new territory).
- **Downstream chooser** (`python -m checks.downstream_check downstream/s0 downstream/s1 --top 0.02`; two
  8-min, 9-worker runs from the first missiles with missile hoarding; 622K decisions, each logged with the
  visits it created; bug fixed on the way: logging a row *view* of the signals table kept every table alive,
  OOM at 16 GB): same logistic chooser, two labels.
  | label | rate | held-out AUC | rarely chosen | holdings | death rate | path length |
  |---|---|---|---|---|---|---|
  | found something new now | 0.60% | 0.957 | +0.81 | +0.01 | -0.11 | +0.15 |
  | created a top-2% lineage later | 0.17% | 0.751 | +0.18 | **+0.11** | -0.08 | +0.01 |
  Trained on long-range outcomes, the learner discovers that holding resources matters (11x the weight),
  without being told; trained on immediate outcomes it learns pure curiosity. Short-horizon power measures
  failed because they asked the immediate question.
- **Chooser race** (from the first missiles, missile hoarding, 2 seeds x 4 workers x 10 min): base 28/39 rooms,
  immediate chooser 72/35, downstream chooser 51/37; no Long Beam in any (budget too small for that door).
- **Multi-horizon, offline** (`python -m checks.horizons_check downstream/s0 downstream/s1`): held-out AUC for
  long-range success: immediate chooser 0.690, downstream 0.751, sum 0.729, learned mix 0.739. Combining
  horizons adds nothing; the long-range label is what matters.
- **Pooled data** (Josh: are we using it fully?): `library.py build` indexes every state any run kept:
  **809,006 states from 150 runs in 28 s** (about 12.5K hold Bombs, about 20K hold an energy tank *without* the
  Long Beam, the Long Beam is not required for the tank). Long-range chooser refit on 8 pooled decision logs
  (1.36M decisions, 4x): held-out AUC 0.751 -> **0.777**; weights: immediate label rarely chosen +0.93,
  holdings -0.03; long-range label rarely chosen +0.17, holdings **+0.09**. Pooling helps the rare-label model.
- Free-resource run from the first missiles (18 workers, 12 min, stop at Bombs): Long Beam, 82 rooms, no Bombs
  or energy tank. The earlier free-resource run from the Long Beam state got Bombs in 416 s; another from there
  did not in 10 min. Even with free resources, Bombs is luck-dependent at this budget.
- **Visuals as generic places** (`python -m checks.visual_check 3000`, 33 s): one frame place per step costs
  **8x** (11,427 -> 1,450 steps/s). Frame places from 3,000 library states (129 rooms): 11x8 / 8 shades (Go-Explore's
  setting): 2,803 places, 97% stay within one room, median 14 per room; 16x15: 2,985 places, 100%; 8x6 / 4
  shades: 399 places, 30%. Pixels carry clean room information but make almost every state its own place
  (enemies, animation, HUD, scrolling); needs masking of moving things and adaptive coarseness (as Go-Explore
  did) and occasional rather than per-step rendering. A real project, not a quick win.
  Coarseness sweep: 8x6/8 shades 10 places per room, 76% pure; 11x8/4 shades 7, 53%; 5x4/8 shades 8, 50%;
  11x8/3 shades 1, 15%. No sweet spot: Metroid reuses tiles across rooms, so coarse screenshots of different
  corridors match. Generic places need a learned representation (e.g. trained to separate states far apart in
  steps) or a better RAM-based one, not plain downscaling.
- **Learned places** (`python -m checks.embed_places_check layer-keep-45min+layer2-watch+paths-15min`: RAM encoder
  trained with temporal contrast, k-means on 3,000 library states): 130/400/1000 groups -> purity 38/51/64% at
  8/10/12 groups per room. Worse than screenshots.
- **Scene-change bytes as generic places** (`python -m checks.scene_bytes_check ...`, 31 s): bytes that change at
  >= X% of scene changes and <= Y% of other steps: (2%, 0.10%): 2,110 bytes, 15 places per room, **91% pure**;
  (5%, 0.05%): 2,048 bytes, 8 per room, **74%**; (10%, 0.02%): 2 per room, 55%. Matches screenshots (76% at 10)
  with no rendering, from RAM, game-agnostic. Metroid's map x/y bytes are excluded (they also change while
  scrolling within a room); room-layout data is picked instead. Missing: generic position within a room.
- **Position bytes found by pushing buttons** (`python -m checks.position_bytes_check 1500`, 29 s): hold RIGHT vs
  LEFT (and DOWN vs jump) for 8 steps from 1,500 library states; rank bytes by how consistently they differ.
  Metroid's x byte ranks 3rd, y byte 1st (byte 20, top on both axes, is likely the controller-input copy:
  needs filtering out). So both halves of a place (which room, where in it) can be found from RAM without
  game knowledge.
- **Do pixels speed up learning?** (`python -m checks.pixel_learning_check layer-keep-45min+layer2-watch+paths-15min
  2000`; 149K steps; held-out AUC; RAM / pixels / both):
  - MLP on 30x32 screenshots: damage soon at 32 recordings .691/.575/.697, at 1600 .863/.797/.879; new room soon
    at 32 .650/.492/.622, at 1600 .746/.703/.763.
  - Small conv net on 56x60 (`CONV=1 SHRINK=4`): damage at 1600 .862/.841/.877; new room at 1600 .743/.728/.768.
  - RAM learns faster at every size; pixels close the gap with more data and a conv; both is best only at the
    largest size (+.015-.025). No shortcut. Reason (correcting my claim): Metroid copies the room layout from ROM
    into cartridge RAM (98% of the found scene bytes are there), so RAM has the geometry.
- **Found places, live race** (`--generic-places`, 6 min x 5 workers from the first missiles): 18/18 rooms, ~18K
  places vs hand-made 39/33 rooms, ~2.7K places. Within one hand-made place the found room bytes take a median 17
  values (object-slot tables refill while scrolling). Choosing bytes by whole-set change rate, or using the
  explorer's own recordings: still 7-13 per room. Stricter scene changes merge rooms (7 bytes: 29% purity).
  No byte subset works because the hand-made "room" (map x/y) itself moves while scrolling: it's a coarse
  position, not a scene id. Next idea: place = bytes moved by holding a direction, each bucketed by how far it
  moves per few steps (multi-scale position), plus the few bytes that change only at the biggest transitions.
- **Found places, more variants** (6 min x 5 workers from the first missiles, 2 seeds; hand-made: 39/33 rooms, ~2.7K
  places; `--generic-places room|position|both`, bytes from `generic_places.py`):
  - direction-moved bytes, top one per direction and hold (too coarse offline): 16/15 rooms.
  - bytes picked by dwell (each changes at most every 16 steps of play, place every 8; hand-made is 8.5):
    drops noise bytes on its own, but live 14/17 rooms, ~17K places.
  - hybrids: found position + hand-made room 23/25 rooms (13K places: splits); found room + hand-made position
    16/7 rooms. Both halves are at fault.
  - carry pairs (`python -m checks.carry_check ...`): fine x/y wrap into 0x30c (a screen bit), not map x/y; Metroid
    moves its map position at screen transitions, not as a byte carry.
  Second-agent review: hand-made "room" is coarse position, not a room id; races with 2 seeds can't separate
  20-30% (same-setup spread 28-39); judge by milestones over more seeds. **Parked** until the Zelda stage; the
  real Metroid bottleneck is long-range (resources paying off far away).
- **Pretrained long-range chooser, live** (`--learned-chooser 1 --chooser-file cache/chooser-later.pkl`, fit by
  `checks.downstream_check ... --save` on 1.36M pooled decisions; fixed, no online refit; 4 seeds x 3 workers x
  8 min from the first missiles): rooms base 28/28/25/27 (median 27.5), chooser 27/31/25/31 (median 29); places
  base ~1.4K, chooser 1.3-2.2K. No acquisition in any run (no tank, no missile capacity, no Long Beam). Small
  edge, inside the noise; this budget is too small to show long-range payoff.
- **Parked idea (Josh): goal-reaching with inflated budgets.** When training "reach G within k steps" from
  recordings, sometimes give a k larger than the recording took, so k means "at most" and the policy doesn't
  stretch to fill it. Pair with shortest-known labels (loop cutting where a state recurs; triangle rule
  d(A,C) <= d(A,B) + d(B,C) over recordings) so it learns the best route rather than copying roundabout ones.
  For the online-agent product; the save-scummer already keeps the shortest replayable path per place.
- **Milestone race** (`tools/milestones.sh`, starts from `near_starts.py layer2-watch`: N steps before Bombs / the
  first tank, 3 seeds x 3 workers x 3 min, plain vs pretrained chooser): Bombs-500 plain 2/3 (162, 172 s),
  chooser 3/3 (106-150 s); Bombs-1000 2/3 (74, 80 s) vs 2/3 (46, 102 s); tank-500 2/3 (12, 20 s) vs 2/3 (6, 14 s);
  tank-1000 0/3 vs 0/3. Total 6/12 vs 7/12: a tie. The chooser isn't the lever. Bombs from 1,000 steps away
  in 1-3 min on 3 workers: this is a usable fast test of hard steps.
- **Autopsy of the tank wall** (`python -m checks.autopsy_check <failed run> layer2-watch tank 1000`: replays the
  known route, finds how far along it the failed archive got and what tries from there did). All 12 failed runs
  (plain, chooser, energy-hoarding) stop at the same spot: route location 31/184, (0,7,17,7,3), the step before
  room (0,7,18). Not energy: the known route ran at 1 energy throughout, and energy hoarding (`--keep-more
  4217,263`) farmed up to 31 energy and reached the spot with 11 but still stopped. Video
  (`videos/known-route-room17.mp4` vs `explorer-tries-room17.mp4`): the route holds right+jump+shoot for ~55
  steps to cross; random tries (repeat 0.9: mean hold 10 steps, 0.9^55 = 0.3%) fall to the bottom and die.
- **Long holds** (`--repeat-probability 0.99 --repeat-spread 0.09`: each try's keep probability drawn from
  0.90-0.99): crossed in 1/3 (to 111/184), with energy hoarding 1/3 (to 119/184); 2/6 vs 0/12 before. Both stall
  next in room (0,7,21). General gap: skills need holds far longer than the explorer's typical ones.
- **No damage** (`--no-damage 263,262`: a try ends like a death the moment energy drops; Josh: humans save-scum
  on any damage) from full-energy starts 1000/2000/3000 steps before the tank (energy edited to 14,
  diagnostic): 0/9 vs plain 0/9. Uninformative: normal holds never cross the room (0,7,17) jump, which comes
  first.
- **Momentum** (`--momentum 0.5`: a try starts by continuing the last button into its visit; idea from the
  second wall, where the furthest visit sits at the exit and tries head back left): with long holds 0/6
  crossed the first wall (long holds alone: 2/6). Machine was overloaded (~46 workers on 24 cores, 25% fewer
  choices); not clean, but no sign of help.
- **Long holds, fair race** (`PARALLEL=2 SEEDS=6 tools/milestones.sh holds "--repeat-probability 0.99
  --repeat-spread 0.09" cache/near-layer2-watch-tank-1000.state`, autopsy for route progress): plain 0/6 past
  route point 31; long holds 2/6 (both to 119/184). All runs tonight: 4/12 vs 0/18. Every crosser stops at 119,
  the exit of room (0,7,21).
- Climb with carry (`climb.py climb-carry --layer0 climb2/layer0 --carry 2`): layers 1-4 from layer 0 events, no
  new acquisitions; first carry step found need 0 at its targets (carried anyway: fixed, carry now skips them).
- **Fast probe** (`python -m checks.tries_check <state> <target room> [tries] [last button]`, ~8 s for 1000 tries;
  `STEPS=` sets try length): from the explorer's furthest saves at both walls, 0% of tries reach the next room and
  ~90% die, whatever the holds; from the known route's save at nearly the same spot, 26-66% reach it (momentum
  best, 66%). With no buttons, the explorer's saves lose energy and die in ~55 steps: **they are sinking in lava**.
  Places can't tell a doomed state from a safe one, and the archive keeps whichever arrived first/shortest.
- **No damage, retested with long holds** from the full-energy tank-1000 start (4 seeds, 3 workers, 3 min):
  long holds 119 / **184 (the tank)** / 31 / 31; long holds + no-damage 31 x4. Over 300 steps 97% of tries take
  some damage (tries_check), so ending tries at damage cuts nearly all long tries. Dropped as a hard rule.
  First tank from 1000 steps away.
- Guard against slow runs: `tools/capped` now kills anything over 4 min unless `LONG="reason"` (logged to
  runs/long-runs.log).
- **Doomed saves, three fixes tried** (long holds, full-energy tank-1000 start, 3 workers x 3 min; doomed = dies
  within 60 steps with no buttons, `python -m checks.doomed_check <room> <runs>`): ~40% of saves kept in room
  (0,7,17) are doomed in every archive, at a median of full energy (14), so energy hoarding can't see them
  (`--keep-more 4217,263`: 0/4 crossed, plain 0/4 in the same batch; 2/4 in an earlier batch: base rate ~25%).
  Up-front check (4 random continuations before keeping): at 30 steps it flags 0/23 doomed saves; at 90 steps it
  catches 19/23 but explorer speed drops 57K -> 4K frames/s. Retiring visits whose tries all die: the lava keeps
  supplying new doomed visits for the same places (87-95% of stored saves doomed, 3-7K places), 0/4. All removed.
  Lesson: stop hill-climbing one optional wall at a ~25% base rate with n=4.
- **Reference chain** (`chain.py`: cheat refills + long holds, stop at each acquisition, continue from the holder):
  step 0 from the Bombs save got the **Ice Beam in 20 s**, then step 1 found nothing in 12 min, 8 rooms
  (342 places, 750K segments, 169 productive). Autopsy: 0/8000 tries from the 8 pocket rooms left them, while
  saves from past runs in the same shaft leave 70-86% of the time. Refill edit and Ice Beam bit don't matter
  (swap tests). Delta-debugging the 1,836 differing bytes found **one byte, 0x49 (scroll direction): 0x00 in the
  trapped saves, 0x01 frees them (43/96)**. The chain reached the Ice Beam through a glitch (wrong scroll mode in a
  vertical shaft); the resulting saves are dead ends. (A side scare: 0x74 reads 0x00 vs 0x10 at the same map
  position, but every 0x00 position also occurs as 0x10, i.e. both are Brinstar as noted; my odd-looking pair was a
  mid-door-transition save.) Fix: pick the continuation save by reachability (quick tries; most distinct places).
- **Reference chain** (`chain.py`: cheat refills + long holds, stop at each acquisition, continue from the holder):
  step 0 from the Bombs save got the **Ice Beam in 20 s**, then step 1 found nothing in 12 min, 8 rooms
  (342 places, 750K segments, 169 productive). Autopsy: 0/8000 tries from the 8 pocket rooms left them, while
  saves from past runs in the same shaft leave 70-86% of the time. Refill edit and Ice Beam bit don't matter
  (swap tests). Delta-debugging the 1,836 differing bytes found **one byte, 0x49 (scroll direction): 0x00 in the
  trapped saves, 0x01 frees them (43/96)**. The chain reached the Ice Beam through a glitch (wrong scroll mode in a
  vertical shaft); the resulting saves are dead ends. (A side scare: 0x74 reads 0x00 vs 0x10 at the same map
  position, but every 0x00 position also occurs as 0x10, i.e. both are Brinstar as noted; my odd-looking pair was a
  mid-door-transition save.) Fix: pick the continuation save by reachability (quick tries; most distinct places).
- **Reference run** (one long cheat exploration from the Bombs save keeping its whole archive: `go_explore.py
  reference --start-state cache/start-bombs.state --refill ... --repeat-probability 0.99 --repeat-spread 0.09`,
  20 workers). Restarting fresh from each acquisition (`chain.py`) lost every healthy frontier: all Ice Beam holders
  came through the scroll glitch and backtracking to them stalled too. Keeping the archive: Norfair + Kraid area
  1.7 min, missile pack 2.7, Ice Beam 3.1, 2nd tank 6.6, Ridley area 9.5, High Jump 11.6, missiles 20 at 12.5;
  then **nothing for 52 min** (225 -> 312 rooms; 24K places; 79K frames/s). Newest rooms (montage) are deep in
  Ridley's area. Self-inflicted wall: the refill set missiles to 5, so no try can beat a boss. Resumed as
  `reference2` with missiles refilled to 99 (`--refill 4218:99,4217:99,...`).
- **Map-guided autopsy** (diagnostic: the world map is a 32x32 table in the ROM at 0x254E; RAM meanings from Data
  Crystal's Metroid RAM map). reference2 (resumed with missiles 99) gained nothing in 45 min and slowed to 28K
  frames/s at 43K places: `Archive.choose` + a full `signals()` table cost 12 ms per try; fixed (cached weights,
  one signals row): 0.2 ms. The run had seen 323/511 map cells and **never a boss** ($6987 "Kraid/Ridley present"
  = 0 in all 43K saves). `checks/frontier_check.py` (quick tries across every explored/unexplored map edge): almost
  all 0 (map neighbours are mostly walls); 2 edges open at the bottom of Ridley's area. **Short explorations started
  at frontiers** (3 min, 4 workers each, `checks/cells_check.py`) covered what the 43K-place archive missed: Ridley's
  bottom row, Kraid's top, and **Kraid's room (29,8)**.
- **Kraid**: 96 HP (enemy slot 0, $040B, mini-boss flag $040F bit 6). Plain exploration from his room: all 48 kept
  saves at full HP; best single try leaves him at 80. Boss damage is invisible to places, so it can't accumulate.
  With his HP in the place (`--place-bytes 1035:8`, diagnostic): saves in every 8-HP band down to 0-7, and **Kraid
  is beaten** (statue byte bit 0, 38 saves) within 3 min. progress.py read bit 7 (statues raised, happens later) and
  missed it; fixed to bit 0. Saved as cache/after-kraid.state. General version to build: find "enemy health" bytes
  automatically (bytes that drop when you shoot, never refill) and add them to places.
- **Ridley** (diagnostic chain): his room is (29,17), found by parsing the ROM's room data (Ridley's room pointer
  table found by scanning his bank for 40 consecutive valid rooms at 0x1618F; the method recovers Kraid's known table
  at 0x121E5; boss entries have type byte high nibble 4: Kraid 0x48 in room 0x1D at (29,8), Ridley 0x49 in room
  0x12; doors: blue from (29,18), 10-missile from (29,16)). (29,18) is a lava corridor: single tries die 96% even at
  99 energy; holding left sinks into the lava below the door. A 3-min exploration with energy refilled to 99
  (`--refill ...,262:0x99,263:0x09`) entered his room. Ridley: slot 0, 140 HP. Fight with `--place-bytes 1035:8`:
  HP bands down to 0-9 in 3 min; resumed with 1035:2: down to HP 1. From HP 1, 12% of 200-step tries start his death
  explosion (slot status 3); the statue byte then becomes **0x02** (Kraid gave 0x01), so a boss kill = statue low
  bits (progress.py fixed again). Surviving post-kill save: cache/after-ridley.state. Both bosses beaten (separately).
- **Tourian** (diagnostic chain). Elevator found by searching the ROM for item entries `04 EE` (elevator, EE =
  destination; the Data Crystal example for Brinstar->Kraid sits at 0x6449): Brinstar entry at 0x63EB puts the
  Tourian elevator at map (2,3), behind the statue room (2,4). With the boss-dead bits edited into a save *before*
  the statue room loads, the statues rise and the bridge forms (single tries reach (2,3) 0.1%; harvested one with
  `harvest.py`); a 2-min exploration from the elevator room entered Tourian ($74 = 0x13). `frontier_walk.py`
  (repeated 2-min explorations restarting at the newest cell) stalled at (7,9): **Metroids latch on and drain ~8
  energy/step** (6 full tanks gone in ~85 steps); HP 255 until frozen. With 6 tanks edited in and enemy slots 0-3
  status ($6AF4+16k, frozen = 4) and HP (bands of 32) in the place, a 3-min exploration got past them to (11,10)
  (151K places). Walk continues from there.
- Tourian walk past the Metroids reached row 11 west to (11,4): **Mother Brain's Zebetite room** (pink mesh
  barriers on pillars, Rinkas). Mother Brain status $98 / hits $99 (dies at 0x20) confirmed from the 6502disassembly
  defines; no Zebetite variables documented anywhere reachable. Two guessed counters (0x63C/0x63D, they dropped while
  firing) change everywhere: not Zebetites. Walk's "newest cell" rule also picks side branches (e.g. (12,10)).
- **Reference reaches the ending** (diagnostic chain, stitched from segments with heavy cheats and Metroid
  knowledge; NOT the method beating the game). Zebetites: no documented RAM; damage-byte finder
  (`checks/damage_bytes.py`: paired tries with/without the fire button; finds Ridley's $040B unaided, misses Kraid's
  because random shots rarely land) found only flags/turret counters there. What worked: `--place-nametable` (crc of
  the background tile layout in the place, so shot-down scenery counts) + `--stay` (tries end on leaving listed map
  cells) in the Zebetite room: through to Mother Brain in 3 min, her hits to 21; resumed 3 min: dying, gone, time
  bomb set ($98 = 6). Escape: up the shaft x=1 from (11,1) to (3,1); the game-end elevator (item 04 8F at (3,1);
  object slot $0320, last byte 0x8F, at x=128 y=131): standing on it at (128,112) and pressing UP -> $6883 = 1,
  credits ($7B) 438 steps later, "The End". cache/ending.state.
  **Walls found, in order** (each needed its own fix): long jumps (hold length), glitched dead-end saves (keep one
  archive), archive bloat/speed (cache choice weights), frontier dilution (short explorations restarted at the
  newest frontier), lava-doomed saves (energy refills), boss damage invisible (enemy HP in place), Metroids
  (enemy status/HP in place + tanks), statues/elevator (boss flags before room load), Zebetites/Mother Brain
  (scenery fingerprint in place + focus), the escape (an elevator random play never stands on long enough).
  General lesson: **every wall after Bombs was "progress the place can't see"** (HP, scenery, object state) or
  "effort spread too thin" (frontier/focus). Next: make both automatic and drop the cheats.
- **General replacement for "boss HP in the place": record places** (`--novel-values`). A save becomes a place
  ("record", room, byte, value) when a rarely-changing byte (changes on < 0.2% of steps) sets a new record low or
  high in this room, with three filters found by autopsy on Kraid (3-min runs from his room, refills on, no hint):
  every new value -> 25K places, Kraid HP 66; records only -> 12K places, 73 (2,171 bytes set a median of 4 records,
  Kraid's HP 33, all downward); one-way only (bytes with records both ways stop) -> 12K, 68; >= 6 one-way records ->
  1.5K from 49 bytes, 52 (projectile positions dominated: table at $05x7); isolated changes only (not changed the
  step before; a hit is one jump, motion is consecutive) -> 209 from 34, 71 (gate too strict now); gate 3 -> 849 from
  139 bytes, **Kraid's HP the top record byte, Kraid to 27**; resumed 3 min (resume now keeps record places) ->
  **Kraid beaten** (1 save), no hand-picked byte.
- Record places replace every hand-made progress hint: **Ridley beaten in 3 min** (start-ridley, refills, no hint);
  **Zebetites + Mother Brain**: from (11,4) with `--stay` and no nametable/MB bytes, through to her room in 3 min,
  resumed 3 min: hits to 28, statuses dying/gone/**time bomb set** (and 7 = bomb exploded: escapes not yet
  survived). Remaining crutches: refills/edited items/tanks/boss flags, `--stay` focus, manual frontier restarts,
  and the game-end elevator.
- **Focus without hints** (from the Tourian entrance, refills on, record places, no stay; 10 workers x 3.4 min):
  recency 0 -> 21 Tourian cells, through the Metroids (no enemy-slot hint) to (11,5); recency 100 (`--recency`,
  newest places weighted up) -> 5 Tourian cells, 201 rooms overall: "newest" includes Brinstar's corners. Recency
  off; record places carry most of the focus.
- **First uncheated run of the method** (`honest1`: from the Bombs save, long holds + record places + missile
  hoarding; no refills/edits/stay; 22 workers x 60 min): Kraid's area 3 min, Norfair 5, missiles 15 at 6, 2nd tank
  13, Ridley's area 40; 287 rooms, 284 map cells; **reached Kraid's room by itself** but no boss beaten. Autopsy:
  the 73 saves in Kraid's room hold energy 0-8 (mostly 3) and 0-4 missiles; Kraid's HP only to 85. The cheats had
  hidden this: **Samus arrives starving**. Energy isn't hoarded (the detector misses the split decimal bytes).
- Resumed with energy hoarded too (`honest2`, `--keep-more 4217,263`, 30 min): archive 176K places (two hoarded
  resources multiply place versions); arrivals at Kraid slightly richer (16 saves at 13 energy, 31 with 5 missiles)
  but Kraid still at 85. The one 83-energy arrival is doomed (0 energy within 10 steps): hoarding keeps it because
  the number looks good at that instant. Honest state: the method reaches Kraid with ~13 energy and <= 5 missiles.
  Next: arrive richer (farm drops / carry riches to where they're needed) and keep doomed saves out.
- **Honest Kraid fight** (healthy honest save in his room: energy 13, 0 missiles; record places; no edits). 5 workers
  x 3 min at four resource levels (real / +99 energy / +15 missiles / both, edited once): Kraid to 74 / 74 / 84 / 76:
  resources barely matter at this budget; the beam alone does damage. 20 workers, 3 + 3 min: 62, then stuck: **every
  lowest-HP save has Samus at 2 energy** (chosen 83-226 times, always dying). Keying records by holdings, and letting
  saves that match a record count: 66 and 78 (n=1 each, no better); both reverted.
- Honest Kraid, 22 workers x 12 min: Kraid to 63 (same as 6 min): more time doesn't help; a mechanism is missing.
- Honest Kraid from a full arrival (energy 199, 10/10 missiles, set once, no refills; 22 workers x 12 min): to 45,
  with Samus at 0 energy and 0 missiles in every lowest save: the fight spends everything (first save to reach each
  record kept, however much it spent). **Records now keep the richest save per level** (a save matching the record is
  offered under the same place, so the keep rule swaps in one holding more): Kraid to 34.
- **Honest Kraid kill**: from a realistic best arrival (2 full tanks = 299 energy, 15/15 missiles, set once, no
  refills; the uncheated run had 2 tanks and capacity 15 by then), record places beat Kraid within 12 min (22
  workers; 609 saves after the kill). The fight is solved; the gap is arriving full (honest arrivals: ~13 energy).
- **Farming works**: from the honest Kraid-room save (13 energy, 0 missiles), 3.4 min x 10 workers of hoarding
  (`--keep-more 4217,263`) -> richest healthy save 1 full tank + 23 (123 energy), 7 missiles; with
  `--favor-holdings 3`: 98 energy, 3 missiles (no better). Target for the kill: ~299 energy, 15 missiles.
- **First boss beaten by the method, no cheats: Kraid.** Chain, all uncheated: Bombs save -> honest exploration
  (`honest1`/`honest2`: long holds, record places, missile + energy hoarding) reached his room -> 20-min exploration
  from a healthy honest save there farmed to 2 full tanks + 99 (299 energy) and 10 missiles but fought from poorer
  branches (Kraid to 50) -> **layer from the richest healthy save** (`pickrich.py`): Kraid dead within 12 min (2,490
  saves after the kill). Loop that should repeat: explore + hoard (farm) -> layer from the richest healthy save ->
  record places win the fight. cache/after-kraid-honest.state.
- **After the honest Kraid kill** (`honest3`: 30 min x 22 workers from cache/after-kraid-honest.state, hoarding +
  record places): richest healthy saves 299 energy + 81 missiles, but **never left Kraid's lair** (52 rooms, area
  Kraid only). Kraid's exit elevator is at (18,7) (item 04 81 in Kraid's bank at 0x12282). Compared with `honest1`,
  which came down via the row 21-22 corridors and the x=1 column, this run never climbs from row 23 to row 22 (quick
  tries from (23,4), (23,7), (23,12): 0/3000) nor west from (27,3) to (27,2) (0/2000; room 0x1A: blue door left,
  red right; screenshots show (27,3) is a vertical two-screen shaft with a solid left wall where Samus lands). No High
  Jump or Ice Beam in these honest saves. Open: the honest route out of Kraid's lair.
- Why the post-Kraid run is stuck: from `honest1` saves the climb (23,7)->(22,7) works 85% (from one save) and
  (27,3)->(27,2) ~4%, i.e. the route exists; `honest3` had only 28 saves in (23,7), all on its right side, and from
  them 0%. The shaft is deadly (60-90% of 300-step tries die). Record places took 58-63% of all choices (`honest1`:
  30.8K of 56.9K places; `honest3`: 13.3K of 42.5K): `choice_group` now counts a room's record places as one place
  when choosing. 10-min check matched `honest3`'s pace (11 rooms at 10 min both); room count was still climbing at
  30 min (11 -> 27 -> 52).
- **Out of Kraid's lair** (`honest5`: 60 min from cache/after-kraid-honest.state, record places grouped per room
  when choosing): 107 rooms, back in Brinstar with Kraid dead, 3rd energy tank, missile capacity 105. Archive 201K
  places (energy x missile variants per place): next stage starts a new layer from the richest healthy save outside
  the lair.
- **Thompson selection** (`--selection thompson|thompson-room`: Beta(1 + successes, prior + failures) per save,
  prior = one success at the archive's rate, or at the room's rate pooled up from the archive's). 10 min, 3-4
  workers, from the post-Kraid Brinstar save and from the Bombs save. Rooms (places): after Kraid: classic 85/75/79
  (mean 80, 3,435 places), thompson-room 78/67/68 (mean 71, 3,079), thompson 66 (1 seed); from Bombs: classic
  63/75/38 (mean 59, 2,653), thompson-room 75/65/39 (mean 60, 2,656), thompson 37. Plain Thompson clearly worse;
  room pooling ties from Bombs and is ~11% worse after Kraid; run-to-run spread (38-75) is bigger than the effect.
  Likely reason: success ("found a new place") is rare (< 1% of tries), so sampling favors a few lucky saves while
  1/sqrt(times chosen) keeps rotating for coverage. Kept classic; option left in.
- **Current honest wall: climbing out of the Kraid-elevator shaft in Brinstar.** Both post-Kraid stages (`honest5`
  60 min, `honest6` 18 min from the richest Brinstar save) only reach the 9 Brinstar cells of the x=6 shaft from (14,6)
  down to the Kraid elevator at (18,6) (plus (18,7), (18,8), (19,7)); the rest is back in Kraid's lair. The first
  uncheated run came the other way: (14,6) is a horizontal room (blue doors both sides, Brinstar room table found at
  0x6324) with a hole in its floor into the shaft; it walked in from the left door and fell. Coming up, tries from the
  top of the shaft reach neither door (0/1200); holding left+fire slides back down to (16,6). The Norfair elevator is
  at (12,22) (replayed from `honest1`'s shortest path into Norfair). Same shape as earlier walls: easy to fall, hard
  to climb back.
- Correction: `honest1`'s shortest route to Kraid's room passes (14,8) -> (14,7) and rides the elevator down; it never
  uses the x=6 shaft. The post-Kraid stages arrive back up into that shaft (by the Kraid elevator) and never get from
  it to (14,7)/(14,8): the "climb back the way you came" picture was wrong; the shaft region's exit is unknown.
  `--climb-records` (new highest point per room in 8-px steps, as its own place): 1,218 climb places in 3.4 min from
  the shaft top, same 8 Brinstar cells as without it (no help; option left in).
- Shaft region doors (Brinstar table 0x6324): Kraid elevator room (18,7) has only a left door into the shaft (18,6);
  (15-17,6) and (19,6) have none; only the top room (14,6) has doors (left to (14,5), right to (14,7)). Extremes
  in any direction as progress (`--climb-records` now: new furthest x or y step per room): 36,853 saves in 3.4 min x 20
  workers, still x blocks 1-6 of (14,6) only, never the door edges. Screens of the highest/leftmost/rightmost saves:
  walls both sides, small platforms, a ceiling of blue blocks, no door visible. Open: whether the doors are on a part
  of the room these saves never scroll to, or the way up is through the ceiling (bombs).
- Frames from a save beside the shaft top's left wall (holding left + fire): a solid pillar, no door; the view is
  scrolled ~121 px into the top room and Samus slides down to (16,6). (13,6) has no room, so the exit isn't up through
  the ceiling: the side doors of (14,6) are above anything these saves see. The climb to the very top of the shaft is
  the wall (`--climb-records` extremes are measured in screen coordinates, which scroll, so they don't capture it).
- **atlas.py**: one screenshot per map cell from any saved run (green border = an uncheated run reached it, orange =
  only cheat runs, grey = on the ROM map, never reached). Uncheated coverage so far: Brinstar 125 cells, Kraid 100,
  Norfair 60, Ridley 12, Tourian 0. It shows the post-Kraid wall plainly: the Kraid shaft (15-18,6) hangs under the
  row-14 corridor, and the uncheated way in was a bombed hole in the corridor floor at (14,6) x~187 (frames from
  honest1). The blocks regrow, so the way back up is a one-way gap for random play. Scripted probes from the highest
  shaft saves (left/right jumps; shoot up then jump, also only under the hole) got 0 out of ~2,600.
- **Post-Kraid shaft wall solved by judging novelty against earlier runs (--familiar).** Autopsy chain: the nesmaps
  Brinstar map shows the shaft top has doors in both walls at corridor height, reached through a stack of blue
  blocks; the uncheated saves do reach a morph-ball tunnel right under them ($0300 = 3, ball), and random tries from
  there leave through a door ~0.4% of the time. But ~90% of all saves were back down the elevator in Kraid lair,
  which a fresh stage archive treats as brand-new, so the tunnel got 37 of 102,487 tries. --familiar runs a,b,...
  starts each new place times_chosen at the choices those runs spent at its location (area, cell, block). From the
  shaft top, 3.4 min x 10 workers: familiar left the shaft in 2/3 seeds (13 Brinstar cells outside it each), plain
  0/3; lair share 71-85% vs 89-92%. Splitting places by pose (--place-bytes 768) alone: no escape.
- **--familiar has the flaw Josh predicted.** After escaping the Kraid shaft, honest7 stalled climbing the column-11
  shaft toward the row-7 door at (7,11)->(7,12). Probes: climbing is easy (16-23% of 100-step tries gain >=48 px,
  4-7% gain >=160 px, at any hold length; absolute height = map row*240 + $FC + (y - $FC) mod 240); the furthest saves
  are ~95 px below the door ledge (the camera, not Samus, was in row 7). But the shaft starts with ~750 familiar tries
  per location (honest1 came down it), vs ~1,225 in Kraid lair, so the route out is marked nearly as stale as the
  basin. Per-location productive rates from earlier runs do not separate them either (0.79% shaft vs 0.82% lair),
  because a new place includes inventory. Counting earlier visits cannot tell a worn-out area from a road that has to
  be travelled again.
- **Familiar history must share permanent progress (Josh's point).** Column-11 climb from its bottom save
  (cache/col11-bottom.state), 3.4 min x 8 workers, 2 seeds: history from post-Kraid runs only put 17-21% of saves in
  Kraid lair and climbed to 1950/2079 (door ledge ~1808); mixed pre- and post-Kraid history 44-47% and 2479/2558; no
  history 60% and 2874/2789. go_explore now counts a --familiar run only if its start save has the same
  permanent_progress (items byte, Kraid and Ridley statue bits). Open: a run that slowly climbed a needed shaft
  (honest7) would mark it stale again; counting visits still cannot tell worn-out from unfinished.
- **Uncheated chain reaches Norfair and Ridley's lair.** From cache/col11-bottom.state (an honest7 save, Kraid dead),
  same-progress familiar history (honest3-6): Norfair within 3.4 min (honest8), 109 -> 153 -> 194 -> 221 rooms over
  3-minute chunks (honest8b-e), then a stall at the Norfair lava corridor (21,26)->(21,25). Autopsy: honest1 crossed it
  on 4 energy without touching lava, so it is technique, not energy; the explorer spent ~1,500 of 3.2M tries on those
  three cells (frontier diluted by ~100K places), and one-cell progress per try is ~1% or less. A fresh layer from the
  richest healthy save at (21,28) (61 energy, cache/lava-edge-farmed.state): Ridley's lair in 3.2 min (lava-layer2,
  64 rooms). The healthy saves at (21,26) itself had 1 energy (a layer from one died at once).
- **Uncheated chain enters Ridley's room (first time any run walked in; the cheat chain had used an edited save).**
  lava-layer2b/c stalled at 95 rooms next to it. Autopsy: (29,17)'s right door and (29,18)'s left door are blue
  (ROM, Ridley table 0x1618F). Probes by ROOM_X were misleading: x wraps inside multi-screen corridors, so "x = 0"
  saves were mid-corridor. Absolute x = map col*256 + $FD + (x - $FD) mod 256 found the nearest saves 36 px from the
  door but under the red-bubble ledge it sits on (nesmaps RidleyHideout.png: image col 0 = map col 14, row 0 = 23).
  A fresh 3-minute layer from the nearest healthy save (cache/ridley-door.state, 1 full tank): 534 saves in the
  room, boss present in 10. Same pattern as the lava corridor: a frontier starved inside a big archive, fixed by a
  fresh layer at the frontier.
- **Ridley fight, uncheated, in progress: 140 -> 115 HP.** Ridley starts at 140 (boss HP in saves with him present).
  Fight layers from in-room saves: 140 -> 119 over 9 min (ridley-door 1-3), 129 -> 115 in 3 min from the lowest-HP
  healthy save (79 energy, 0 missiles). Too slow and energy-limited. Missiles are the lever but scarce: best in the
  room 8; farming 3 min in the row-30 corridor raised the best healthy save from 4 to 25 missiles (394 energy, at
  (30,16), cache/ridley-farmed.state), but a layer from it reached the door corridor (29,18) only with 0 missiles and
  11 energy (ridley-carry): the trip spends what farming built. Farming right outside the door gave 1 missile, 103
  energy. Wall: arrive rich (carry riches to where they are needed). Resumed runs keep no button paths, so routes
  found inside resumed archives cannot be replayed.
- **Ridley beaten by the method, no cheats.** Fight layers from in-room saves took him 140 -> 109 HP slowly (~6 HP per
  3-min layer). Probe: standing still facing him, holding LEFT+B for 60 steps does 1 HP; pressing B every step does 6.
  Most games act once per press, so long holds fire once. Fix (general, any NES game): Emulator.run understands a
  TURBO bit (A and B pressed for the first half of the step's frames only), and controls.ACTIONS gets turbo copies of
  every action with A or B, appended last so old stored paths keep their meaning (env TURBO=1 for now). Race from the
  109-HP save, 3 min x 10 workers: turbo 109 -> 78, plain 109 -> 89 (one seed). Ratchet (ratchet.py: next layer from
  the lowest-boss-HP healthy save with >= 60 energy) with turbo: 78 -> dead within 3 min (ratchet-3, 1,984 saves
  with Ridley dead). Chain so far: Bombs -> Kraid -> Kraid shaft -> Norfair -> lava corridor -> Ridley lair ->
  Ridley's room -> Ridley, all uncheated; hand steps: choosing each layer's start save.
- **After Ridley, uncheated: out of his lair, back through Norfair (High Jump picked up on the way), into Brinstar,
  and the Ice Beam.** Every step was the same move: a 3-minute layer from the healthy save nearest the stalled
  frontier, with same-progress history. Stalls and their causes: the lair exit door behind a step of grey blocks
  (25,26); the Norfair column-30 climb; the (10,22) bricks that seal the Norfair elevator room off from the shaft above
  (the Ice Beam route comes down column 22 from the row-7 corridor, nesmaps map); the column-18 shaft doors on row 7.
  ice3 (from cache/col18-door.state): Ice Beam within 3 minutes.
- **Uncheated chain reaches Tourian.** From the Ice Beam save (0 missiles, 164 energy): honest11 a-c (3-minute
  chunks, turbo) farmed to 19-21 missiles while wandering Brinstar, stalled with the column-11 shaft topping out in
  row 3; a layer from the highest healthy column-11 save with >= 5 missiles (3 energy!) went up, left along row 2,
  through the statue room and into Tourian within 3 minutes (tourian-approach).
- **Uncheated chain through Tourian to the Zebetite room.** Missiles were the bottleneck after the statue room
  (every Tourian save had 0-6). Random play wastes some (0.8-0.9 missiles per 100-step try in a quiet room; missile
  mode on 21-33% of the time) but the main cost is the Metroids along row 7. Josh's point held: **killing Metroids
  drops lots of missiles** - a 3-minute layer from an 18-missile save in the first shaft reached 80 missiles, and a
  layer near Mother Brain went back up the column-10 shaft and farmed to 150. The orange door at (7,9) -> (7,10) is
  the 10-missile door; past it the column-10 shaft drops to row 11 and the corridor leads left to the Zebetite room
  (11,4) (nesmaps Tourian.png: image col 0 = map col 1, row 0 = row 3). pick_layer.py now picks layer starts (healthy,
  best missiles/30 + energy/300 balance) after several bad hand picks (1-5 energy).
- **Zebetites: current wall.** Mother Brain status 0 in every save. Runs stop at the room's left part; "(11,3)" saves
  were camera scroll. Missiles fire about one per 8 steps (one in flight at a time; turbo does not speed them up),
  Rinkas take ~30 energy per 24-30 steps. 120 steps of scripted fire at the first Zebetite: 11 missiles hit, 60
  energy lost, Zebetite still standing. Race, 3 min x 10 workers, background fingerprint (--place-nametable) on vs
  off: neither got past. absx() is unreliable in this room (x=1 saves are at the right door).
- **Zebetites passed and Mother Brain hit, uncheated (diagnostic --stay still in use).** Scripted probes: Samus is
  stopped by the pillar the Zebetite sits on; missiles fired from the blue platform at Zebetite height chip it
  (stripe pixels 31 -> 10), but Rinkas knock her off within ~40 steps. --place-nametable alone (3 min race): no
  progress. --place-nametable + --stay "11,4;11,3;11,2;11,1" (hand-listed cells, diagnostic only, to generalize):
  through to Mother Brain. Resources decide it: the best (11,2) arrivals had 56 missiles / 59 energy; from there
  Mother Brain's glass broke (status 2) and hits reached 6 of 32 (mb-stay6/7), ending with 12 missiles / 29 energy.
  Missiles and energy never peak in the same save; best joint stock found: 91 missiles / 149 energy at (11,8)
  (Metroid shaft drops), 108 / 179 at (11,4). Near Mother Brain every save fails the idle-health check (turrets and
  Rinkas drain), so the ratchet picker fell back to poor saves.
- **GAME BEATEN by the method, no cheats (2026-11-26).** No edited RAM in any link of the chain (only diagnostic
  probes used edits, never fed back). Mother Brain: rich arrival from farming the column-10 Metroid shaft (127
  missiles / 284 energy, stay-limited farm layer), crossing to (11,4) with 126 / 308, then ratchet layers from the
  most-hits save (pick: hits, then missiles/60 and energy/200): hits 9 -> 17 -> 24 -> 30, then a layer from a 29-hit
  save (7 missiles, 32 energy) killed her ($98 = 6, time bomb set). Escape layer (no stay): up the x=1 shaft to the
  game-end elevator, $6883 = 1 in 128 saves, credits in 4 (cache/ending-honest.state; items 0x95, statues 130/130).
  **Hand-made parts that still need generalizing:** choosing every layer's start save (pick_layer / pick_joint /
  mbr / ratchet scoring), --stay cell lists in Tourian (Zebetites, Mother Brain, the Metroid farm), the Metroid-specific
  hand-made place and energy/missile bytes, TURBO as an env flag, and the walkthrough/map knowledge I used to aim
  autopsies (never fed to the explorer).
- **autochain** (no hand-picked saves): from the post-Ridley save, back to Brinstar in 15 x 3-min chunks (138 rooms) vs 6
  chunks with hand picks. First version relayered from the best-holdings save among places added in the last chunk:
  it picked Ridley lair (wasted 3 chunks). Fixed rule: healthiest best-holdings save in the most recently discovered
  room (offline check on the stalled archive: lands at (21,29), next to the hand pick (20,30)). Remaining slowness:
  the "no new room" stall test never fires while side rooms trickle in.
- **2026-11-27, laptop (unit offline).** No general feature recovers the 20 hand-picked layer starts (best: fewest
  saves in the room ranks the pick at the 22nd percentile; newest room / most holdings do worse than random; good picks
  often had 1-14 energy). Measured usefulness doesn't work as a label either: random tries from one save (400 and 3000
  steps, incl. the hand picks) never broke a wall in 13 archives; walls fall to chains of saves, not single tries.
  Next label: ancestors of the first success in fresh runs (the stepping stones), free from stored parents.
  heuristic/ (laptop) has the scripts. Speed: shutdown 55.7 -> 1.6 s per run, history cache 70.6 -> 1.5 s,
  weights() 72 -> 27 ms at 62K visits; reproducible runs (worker seeds by index, identical fingerprints across
  machines); --max-frames budget; cleanup redone (go_explore + explorers 898 -> 604 lines).
- **Stepping stones** (ancestors of the first success, 21 fresh bench runs): chains of 6-25 saves; they got 1-17%
  (typically 2-9%) of all tries, ~1.5-2x their fair share. So >90% of tries go to saves that never lead anywhere: the
  upper bound for a better chooser. Feature AUC for "is a stone": children created 0.89, times chosen 0.62,
  found-new rate 0.61, holdings 0.51, path length 0.34 (stones are early), deaths/try 0.21 (stones die more), record
  place 0.01 (never). heuristic/stones/.
- Rejected, 3 walls x 3 seeds at a 3M-frame budget (laptop, 8 workers): --stale-steps 40 (end tries after 40 quiet
  steps): no better, worse on lava. --fertility 1 (weight by 1 + children): worse on all three (e.g. Ridley door
  0.6/1.1/0.2M -> 2.2/2.1/0.5M); children are a consequence of being a stone, so weighting by them is rich-get-richer.
- **Breadth test, simple/explainable ideas** (offline, 30.7K states / 115 rooms from 11 runs; 30 runs for frontier;
  heuristic/breadth/). Rooms (purity / ids per room): scene-change bytes refit 70% / 9.3 (best); tile bag 43-52% /
  3-5 (tiles reused across corridors, columns stream in); scene cuts + tile bag 64% / 2.1 (RAM bursts of 300+ bytes
  mark doors; the nametable is rewritten column by column, never at once); save-tree clusters 63-75% / 23-31.
  Background palette and a CHR-RAM checksum: 100% pure AREA ids, free. Not room-based: slow bytes give one knob from
  ~4 to ~360 effective ids per run. Frontier (stone AUC): descendants 0.97 (0.94 using only info known 25% into the
  run), children 0.93, found-new rate 0.74, "generations since the tile bag last changed" (fewer = higher) 0.73 and
  0.75-0.78 for the first success itself (no hindsight); zlib novelty 0.50-0.57 (stones are old hubs, not novel
  states). Note --fertility (children) already lost a live race; to race: milder descendants weight, and
  generations-since-scenery-change.
- **Learned step-distance model** (heuristic/distance/; 161K replayed states from 48 fresh bench runs; inputs: 3,430
  bytes that vary but aren't timers, value + "differs from usual"; MLP 6860-512-256-32, distance trained to log(1+steps
  apart), 156 s on CPU). Held-out distance correlation 0.94. Rooms: 92-95% purity at 5-10 places per room, but an
  untrained random encoder (90-94%) and PCA of the raw inputs (92-95%) do as well on these ~8-room runs: no gain.
  Stepping stones: every learned novelty score at or below chance (0.32-0.55); even a cheat (closeness to the actual
  first success) only 0.65 - stones' siblings are just as close. Not wired in. The frontier isn't distance/novelty.
- **One-way moves** (heuristic/oneway/; 18 successful bench runs, stones + 60 other saves each, 10 random tries of
  150 steps from each save: does any get back to the parent save's location?): one-way AUC 0.39 - stones are
  slightly MORE reversible (return rate ~0.27 vs ~0.27-0.30, varies). Hypothesis rejected. Converging picture from
  all frontier tests: stepping stones are old, central, well-connected hubs (early, many descendants, die more),
  not novel states or one-way edges.
- Memory check for the unit crash (laptop): resumes of 8K / 36K-place archives peak at 0.9 / 2.5 GB; workers stay
  ~50-63 MB; main grows ~55 KB/place. A 100K-place resume should need ~6-12 GB, far under the 40 GB cap, so memory
  is an unlikely cause.
- **Selection weights from hindsight signals, raced** (laptop, 8 workers, 3 walls x 3 seeds, 3M-frame budget; frames
  to break ridley-exit / ridley-door / lava-corridor). Control rerun on current code: 0.6/1.5/0.6M, 1.1/1.2/0.6M,
  2.5/1.3/1.4M; the first control (0.5/0.2/1.0, 0.6/1.1/0.2, fail/1.5/0.6) was lucky. Vs pooled control (6 seeds):
  --fertility 1 clearly worse; --descendants 0.5 and --scenery-recency 1 about even on exit and lava, worse on the
  door (~1.8M vs ~0.8M); --stale-steps 40 about even, worse on lava. None kept. **Noise lesson:** per wall, seeds
  spread ~+-0.5M frames, so 3 seeds only detect ~2x effects; smaller real gains need many more seeds or walls.
- **autochain, no-picking (MULTISTART) vs newest-room pick** (laptop, post-Ridley save, 6 workers each, 24 x 3-min
  chunks, ~3M frames per chunk = ~72M per arm, about ONE unit chunk; laptop is ~7x slower per worker): neither left
  Ridley's lair. Newest-room pick: 9 layers, rooms reset each layer (8-26). Multi-start (new layer = every save the
  stalled chunk found): 12 layers, rooms grow steadily to 48 - it never throws away the frontier. Structural win at
  equal compute, and it removes the hand pick. Continuing the multi-start chain with 12 workers (ac-multi2).
  Note: the stall test counts record places, hundreds per chunk, so it fires about every other chunk.
- **Multi-start chain continued** (ac-multi2, 12 workers, ~6M frames per chunk; ~156M frames total incl. ac-multi):
  rooms 48 -> 49, then flat for 6 chunks; reached (25,27), one cell short of (25,26), the grey-block step before the
  Ridley-lair exit door that also stalled the hand-run chain (which used ~144M frames to reach (25,26), then a layer at
  that frontier). So no-picking got about as far on similar compute, then stalled at the same wall. The stall test is
  noisy (record places count as discoveries), so layers restart every other chunk even while rooms are flat.
- **Laptop throttled:** Ryzen 7840HS at ~950 MHz under load (below its 1.1 GHz minimum; boost on, performance
  profile, 41-44 C, 65 W charger) => ~3K emulator frames/s per process vs ~20K per worker on the unit. Hardware/
  firmware throttle (EC/PROCHOT); fixing it could give the laptop ~4-5x compute. Needs Josh.
- **No-picking autochain leaves Ridley's lair** (ac-multi3, MULTISTART + rooms-only stall test, 12 workers): rooms flat
  at 50 for 9 chunks with a new layer every other chunk, then 55, 57, and Norfair at chunk 12. Total for the stretch
  (ac-multi + multi2 + multi3): ~234M frames, vs ~216M for the hand-guided chain (honest9/9b + ridley-exit layers).
  Same cost, zero hand picks. Continuing toward Brinstar.
- **No-picking chain, end of night** (ac-multi3 finished 30 chunks): Norfair, 90 rooms, not yet Brinstar. Rooms keep
  climbing slowly (63 -> 90 over 18 chunks, ~108M frames). Whole stretch from the post-Ridley save: ~340M laptop
  frames, zero hand picks. Next: resume it (CONTINUE_RUN=ac-multi3/c29) on the unit, where it runs ~7x faster.
- **No-picking chain plateaus in Norfair** (ac-multi4, resumed from ac-multi3/c29, 38 chunks x ~6M = ~230M frames):
  rooms flat at 90-93 the whole run; a stall fires every other chunk and each new layer (3-3.4K saves) re-finds the
  same rooms. Note "rooms" counts only the current layer's archive, so it can't grow once a layer covers the area.
  Open question for next time: is it genuinely stuck (which Norfair wall?), or is the stall-and-restart loop itself
  wasting the compute? Cheap check: map which cells the layers reach vs the known route to Brinstar, offline.
- **Why the no-picking chain stalled in Norfair** (offline check vs honest10e, which went Norfair -> Brinstar):
  the chain reached the Norfair shaft at (19,30) but never (18,30), the next cell on honest10e's route up. Its last
  chunk spent 91% of tries (15,194 of 16,738) in Ridley's lair, an area already done, because a multi-start layer
  carried every save, and only 0.7% at the shaft, from saves with 0 tanks and 0-2 missiles. Diagnostic: resume only
  its 1,409 Norfair saves -> (18,30) reached in 3 min / 5.8M frames (the chain: never, in ~230M). So the "wall" was
  wasted attention. General fix (31b78e0): a new layer keeps only saves in rooms the stalled layer discovered
  itself. Testing from the post-Ridley save (ac-front).
- **Correction: the laptop isn't throttled at rest; it has a load cliff.** 1/2/4 emulator processes: ~4.5 GHz,
  30K/59K/117K frames/s. 8 processes: all cores ~1 GHz, 47K total. Overnight's 12-worker runs got ~33K, so every
  laptop chain last night ran at about a quarter of what 4 workers would have given. Suspect a 65 W charger limit.
- **Thinkpad (joshm-thinkpad-1) online:** reproduces refA exactly; 16 workers ~172K frames/s.
- **Frontier autochain reaches Brinstar** (ac-front2, thinkpad, 15 workers): post-Ridley save -> Brinstar in 23 chunks,
  ~430M frames, zero hand picks, once "discovered" meant new to the whole chain (bc10a62). The layer-only version
  kept re-finding Ridley's lair (90% of tries there) and never left Norfair. Paused at chunk 26 for the place bench.
- **Generic places, offline (honest10e saves; 'aliased' = a place holding 2+ hand rooms, a diagnostic only):**
  screen cells alone have no sweet spot: 16x12x8 / 11x8x8 give ~1 place per save (1-4% aliased); 8x6x4 / 4x3x4
  give 560-800 / 80-115 places but 75-83% / 98% aliased. Slow RAM bytes (auto-chosen by change rate) are worse:
  hundreds of slow junk bytes make every save unique (2400-2900 cells per 3000 saves). Context (Josh's idea: a
  place is defined by its neighbours): coarse cell + previous 1/2/3 different cells cuts 8x6x4 aliasing 91% ->
  76/66/56%. Fine cells as "words" fail: 30K cells in 90K steps, each seen ~3 times, so context can't repeat.
- **--place-scheme screen:W,H,D[,N]** (c3ea262): shrunk gray picture (every 4th pixel, last frame of each step
  drawn; 0.04 ms) + N previous different cells. Hand scheme unchanged. Bench: hand vs screen:11,8,8 vs
  screen:6,4,4,2, 13 walls, 10M-frame cap, history off, seeds 1-5, on the thinkpad.
- **Place bench round 1** (seed 1, 10M-frame cap, history off): hand 4/13 (lava 0.9M, ridley-door 1.1M,
  ridley-exit 0.7M, missile-door 5.7M; the other 9 need history, --stay or >10M). screen:11,8,8 = every save new
  (290-400K places in 10M frames) -> hit the 24G cap and crashed. screen:6,4,4,2 = nothing new (358 places in the
  lava run) -> 0/13. Granularity dominates; sweeping between them.
- **Hardware scroll is readable generically:** QuickNES state block PPUR bytes 8-9 = the PPU's 15-bit temp address
  ("loopy t": coarse x/y, nametable, fine y), byte 10 = fine x. Decoded scroll matches Metroid's own $FD/$FC 73% (x)
  / 87% (y) of 3077 saves (misses likely frame timing). Basis for a human-like place: stretch between screen
  transitions + scroll travelled within it (dead reckoning), with the previous stretch as context.
- **Plain screens fail at every fixed grid** (seed 1, 4 walls hand breaks): best screen:8,6,4 broke 1/4 (ridley-exit
  2.6M vs hand 0.7M); context variants and 11x8x4 broke 0/4. Place counts swing 589 -> 14,077 between walls for one
  setting: a fixed grid is coarse in one area, fine in another.
- **Dead reckoning (scroll:B) is the first generic scheme that works** (seed 1): lava 1.4M (B=8; hand 0.9M),
  ridley-exit 1.1M (B=16; hand 0.7M), missile-door 8.9M (B=16; hand 5.7M); ridley-door never (hand 1.1M). Place
  counts stay sane (640-12K). Missing: where the player is on screen (height/aim) - next, find player-position bytes
  by contingency (which bytes respond to the controller).
- **Contingency finds the player** (`python -m checks.contingency_check runs/honest10e/checkpoint.pkl.z`): score =
  bytes whose right-minus-left (jump-minus-nothing) difference has a consistent sign across 300 saves AND grows from
  3 to 6 steps of pressing. Samus x ($30E) ranks 2nd of 2048+8192 bytes for right/left, Samus y ($30D) joint 1st for
  jump; the bytes beside them (81, 82, 6-9) look like working copies. Without the "grows" test the controller byte
  and facing flags crowd the top. Plan: probe at run start, add the top bytes per axis (in blocks) to scroll places.
- **Place bench seeds 1-3** (4 walls, 10M cap): hand 10/12, scroll:16 6/12, scroll:8 5/12 (2-5x hand frames when
  they pass; scroll:8 broke ridley-door once, 6.0M). Seed 1 had flattered scroll.
- **Camera moves from pixels alone** (`python -m checks.pixel_scroll_check runs/honest10e/checkpoint.pkl.z`): matching
  the column and row brightness profiles of consecutive frames gives the hardware scroll change exactly in 98% of
  1,387 moving steps (100% of 4,474 still ones). So dead reckoning needs no NES hardware: only "how did the view move".
- **Contingency player bytes in places** (`scroll:B,P`, 15f5f5e): the run-start probe picks Samus x/y (781/782) or
  their copies from 3 of 4 starts (lava edge picks two other bytes for the across axis). P=8 is too fine (1-3.6K
  places per 300 tries vs hand 177); benching P=16/32.
- **2M-frame cap, 5 seeds, 4 walls** (Josh: tight budgets force smarter search): hand 9/20 (lava 1, ridley-door 3,
  ridley-exit 5, missile 0); scroll:8,32 6/20 (ridley-door 5/5 at median 0.8M, beating hand); scroll:16 2/20;
  scroll:16,32 1/20; scroll:8 0/20.
- **The bench is a coin flip per wall at this size:** scroll:16,32 on ridley-door went 0/5 in that bench and 4/4
  in a rerun with byte-identical config.json (same player bytes 782,81,781/82). Failing runs wander into other
  rooms (4 rooms, 60 layouts) and spend the budget; passing ones open the door by ~0.5M. 15-worker runs aren't
  reproducible (worker timing), so seeds are independent draws. Per-wall verdicts need ~10+ seeds; only large
  totals mean anything at 5.
- **compare.py, hand vs scroll:8,32** (15 paired seeds, 1-worker runs, 2M cap): hand 2.33 walls/seed, scroll 1.07
  (A-B +1.27, 95% +0.93 to +1.60). lava 7/15 vs 0/15, ridley-door 14/15 (0.51M) vs 12/15 (0.70M), ridley-exit 14/15
  (0.61M) vs 4/15 (1.61M), missile-door 0 vs 0.
- **One worker beats fifteen per frame:** hand broke ridley-door 14/15 with 1-worker runs vs 3/5 with 15-worker runs on
  the same 2M budget. Each try sees the archive as of the previous try. Many small parallel runs > one wide run.
- **Restarts rejected (at <=2M):** expected frames per pass with restarts every c = mean(min(f,c))/P(f<=c), from the
  15 seeds per wall above: never better than one full run beyond noise (lava hand 3.5M at c=2M vs 11.1M at 0.75M;
  ridley-exit 0.91M vs 1.39M at 0.5M; ridley-door 0.67M vs 0.62M best). Time-to-pass isn't heavy-tailed; runs build on
  their own archive.
- **Round 1 add-ons, batch 1** (15 seeds, vs scroll:8,32): poke +0.07 (-0.33 to +0.47; ridley-exit median 0.97M vs
  1.61M), area 0.00, burst -0.07, fuzzy -0.40, sticky -0.47, cast -0.47, cuts -0.80 (-1.13 to -0.40), patch -0.87
  (-1.20 to -0.53). No generic scheme breaks the lava corridor (hand 7/15).
- **Round 1 was OOM-killed at its 24G group cap** after 2.5 h (one exploding run took all 15). compare.py now caps each
  run (1.5G; a killed run counts as not breaking the wall). The watcher only woke on summaries, so it sat dead an
  hour: watchers now also wake on no process / 15 min without a finished run (Josh: check work, don't wait).
- **Why hand wins the lava corridor:** not inventory (hand without items/tanks/missiles: identical 7/15), not block
  size alone (scroll:32,32 0/15, scroll:8,32 1/15). One run each: hand kept 1,077 saves over 20 map cells and went
  down the shaft at column 24 into Ridley's area; scroll:8,32 kept 6,171 saves in 5 corridor cells (~1,500 per cell
  vs ~100) and never reached column 24. Too fine again: the 4 contingency bytes multiply with the 8 px blocks.
- **Round 1 verdicts (30 paired seeds vs scroll:8,32; walls/seed):** worse = sticky -0.70, cuts -0.80, cast -0.53,
  fuzzy -0.47, patch -0.80 (all intervals below 0); no difference = area 0.00 (+-0.20); undecided = poke 0.00
  (+-0.27; ridley-exit median 0.97M vs 1.59M), burst -0.13 (-0.30 to +0.03). Every loser adds fast-changing detail
  (enemy sprites, hash bits, timer bits, local tiles) -> finer places, the same failure as the lava corridor.
- **Screen of 12 ideas** (8 paired seeds vs scroll:8,32; ridley-door + ridley-exit; walls/seed, map cells/run):
  clearly worse on both scores: afford -1.38/-2.5, reach -1.12/-2.9, oneway -0.62/-1.4, split50 -0.88/-3.3,
  split200 -0.75/-2.9 (probes pay their frames; crowding splits end too fine). Level or better on something: player1
  0.00 (ridley-door median 0.43M vs 0.98M), radius24 -0.12/+0.9 cells, radius12 0.00, groups -0.38 (best ridley-exit,
  4/8), stitch -0.38/+0.5 cells (0/8 ridley-exit). Leaning worse: edges, levels (fine 4-32 px).
- **Coarser places, laptop, 4 seeds vs hand:** scroll:16,64 player-per-axis 1 matched hand on lava (1/4 each; first
  generic lava passes) but 2/4 vs 4/4 on ridley-exit; scroll:32 without the player part -5.1 cells/run. The player
  part is essential; lava wants coarse, ridley-exit finer -> motivates every-grid-at-once spanning coarser sizes.
- **Search re-tests on hand places** (paired, ridley-door + ridley-exit): at 4 seeds all looked >= hand, at 12 seeds
  hand 23/24, fertility 20/24 (-0.25, -0.58 to 0.00), scenery recency 20/24 (-0.25, -0.67 to +0.17); descendants
  level at 4 seeds. No sign of gain; the old rejections stand in spirit (intervals still touch 0, so not "worse").
- **Ridley-exit diagnosis** (one run each, seed 2): hand kept 1,083 saves, tried (24,25) 82x and (23,25) 16x, reached
  Norfair; scroll:8,32 kept 10,423 saves (~10x), tried (24,25) 410x but (23,25) once, never left. Same failure as the
  lava corridor: generic places ~10x too fine. Next: screen-sized dead-reckoning blocks (256/128/64 px) + player
  bytes in 32 px steps, the generic mirror of hand places (screen + position on it); possible now that saves carry
  exact positions.
