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
