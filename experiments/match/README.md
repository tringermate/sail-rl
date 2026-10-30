# Match racing from a situation library (ideas.md, 29/10/2026)

A 1v1 policy trained on short, decisive encounters instead of only full races, in the spirit of SGS's fixed
set of task configurations (s0, g, e). Each training entry is a **setup** (two boats placed on the course) x a
**role pairing**. Question: does a policy trained on situations beat one trained on full races with the same
compute, *in full races*?

## Rules check (sail-env), and what changed

The core of the Racing Rules of Sailing that sail-env implements is correct: RRS 10 (port/starboard), 11
(windward/leeward), 12 (clear astern, by the definition's line abeam the stern), 13, 14 (contact costs), 16.1
(optional `rrs16`), 18 (simplified: 3-length zone, room decided when the first boat enters, inside overlapped
or clear ahead entitled), 22.2 (a penalised boat keeps clear), 31 (touching a mark), 44.2 (get clear, then
turn). Three deviations mattered for 1v1 situations. Two are now fixed as **opt-in flags** (default off, so
earlier presets and results are unchanged; tests in sail-env):

| | sail-env 0.1 | real rules | now |
|---|---|---|---|
| gybing boat | keeps clear like a tacking boat | RRS 13 is tacking only | `rrs13_tack_only` |
| port/starboard at the windward mark | the inside boat gets mark-room across tacks | RRS 18.1(a): no mark-room between opposite tacks on a beat (RRS 10 decides) | `rrs18_opposite_tacks`, with a simplified RRS 18.3 (a boat tacking in the zone owes room) |
| acquiring right of way, proper course | not modelled | RRS 15, 17 | approximated by `rrs16`: a right-of-way boat that turned > 10 deg in the 2 s before a contact is at fault |

Not modelled and accepted: RRS 15 and 17 beyond that approximation, 16.2, 19 and 20 (obstructions), match-racing
Appendix C (umpires, entry rules, penalty with a tack and a gybe). A penalty here is one 360-degree turn after
getting clear. All configs here switch on `rrs16`, `rrs13_tack_only` and `rrs18_opposite_tacks`.

## Situations

`situations.py`: 12 situations. L = boat length (6.9 m), distances in metres unless in L. Boat 0 is A or B at
random, so the index says nothing about the geometry.

| situation | setup | target |
|---|---|---|
| prestart_entry | T-80..T-55: A near the pin end (starboard, TWA 60-110), B near the committee end (port), 40-90 m below the line | legal start, then 50 m upwind |
| start_leeward | T-30..T-12: both on starboard 15-45 m below, B 1-3 L to windward, overlapped | as above |
| start_chase | T-50..T-25: B 2-5 L astern of A, same course +-15 deg | as above |
| port_starboard | beat, 60-160 m below the mark: close-hauled on opposite tacks, collision course, B crossing up to 1.5 L ahead or behind | round the windward mark |
| same_tack | beat: same tack, B within +-4 L fore-aft and 1.2-5 L to windward or leeward | round the windward mark |
| cover | beat: A 3-8 L straight upwind of B (B in dirty air), tacks at random | round the windward mark |
| wmark_port_starboard | A on the starboard layline 3-8 L out, B on port crossing near the mark | round the windward mark |
| wmark_overlap | both on the starboard layline, 4-9 L out, B overlapped inside/outside or just astern | round the windward mark |
| run_blanket | run: B 2-6 L upwind of A (can blanket), gybes at random | pass and round the gate / finish |
| run_converge | run: opposite gybes, collision course | pass and round the gate / finish |
| gate_approach | 4-10 L above the gate, overlapped or just astern | pass and round a gate mark |
| finish_approach | 4-10 L above the finish, overlapped or just astern | finish |

Beat and run situations are on the first or second lap at random (the race clock matches roughly). Speeds:
60-100% of the polar. Every setup also gets **t_solo**, each boat's undisturbed time to the target: the
hand-written digital sailor sails it alone (the other boat removed). For starts it is projected instead (reach
the line at beam-reach speed, not before the gun, then 50 m at best VMG): from late approaches the digital
sailor first goes to its own spot on the line and starts 20-80 s late.

## Goals and reward

Three role pairings per setup: **race vs race**, **block vs escape**, **escape vs block**.

- **race**: first to the target wins (with no penalty left to serve). Nobody there after 2 x the longer
  t_solo: less distance still to sail wins.
- **block / escape**: the escaper wins if it reaches the target within `block_factor` x its t_solo (for a
  start, `block_factor` x its time after the gun), else the blocker wins. The blocker's own progress does not
  count. `block_factor` = 2 ("about double the projected time").

Reward: +1 / -1 / 0 at the end (zero-sum), sail-env's contact costs, -0.5 per penalty incurred (so a
deliberate foul to block does not pay), and potential-based shaping (own distance to sail; the blocker's
potential is the opponent's distance to sail) faded out over training. One shared policy sails both boats; it
observes its role and the opponent's, the time left and whether the target is the start checkpoint (8 inputs
after the race observation, `env.py`). 5% of episodes are full races (about a third of the steps).

**Curriculum signal.** In self-play the two boats' outcomes always sum to zero, so the sampler records boat 0's
outcome only: per setup and pairing, how often the side boat 0 is on wins. SGS then favours setups where that
side wins about half the time. This measures how balanced a setup is, not learning progress as in the SGS
paper, and that is the main reason not to expect much from M2 over M1.

## Experiments

| config | what |
|---|---|
| `m0_control` | control: full 1v1 races only (same env, inputs, rewards, compute) |
| `m1_uniform` | situations, uniform sampling |
| `m2_sgs` | situations, SGS (paper defaults, as p3j_sgs) |
| `m3_sgs_staged` | + staged success: `block_factor` 1.3 -> 2.0 over the first 60% (the success curriculum that won at starts) |

All: warm start from the bundled racer, 2048 envs, 200M steps per seed, 3 seeds, half the envs against 4
frozen past selves. Decisive measure: **full 1v1 races, M1-M3 vs M0, head to head**; per-situation tables and
GIFs to see *how*.

## Run on Runpod

On the laptop: commit both repos first (pack.sh packs tracked files at their current contents; new files must
at least be `git add`-ed), then as in `experiments/runpod/README.md`:

    bash experiments/runpod/pack.sh
    # on the pod
    bash experiments/runpod/run_queue.sh experiments/configs/match/m0_control.yaml experiments/configs/match/m1_uniform.yaml \
        experiments/configs/match/m2_sgs.yaml experiments/configs/match/m3_sgs_staged.yaml

Before the queue: a 1-minute check that the pod runs it at all (`--total-timesteps 200000`), and the throughput
(`num_envs`). Building the bank takes ~2 min on a laptop CPU for 16 setups per situation; at 512 it is one jitted
batch on the GPU.

    python -m experiments.match.train experiments/configs/match/m1_uniform.yaml --no-wandb --total-timesteps 200000

Evaluate (held-out bank: bank seed + 1):

    python -m experiments.match.evaluate runs/train/m1_uniform_<stamp> --vs runs/train/m0_control_<stamp>
    python -m experiments.match.evaluate runs/train/m1_uniform_<stamp> --vs self --gifs 1   # block/escape within one policy
    python -m experiments.match.evaluate runs/train/m1_uniform_<stamp> --vs digital

Live curves: `block_success` overall and per situation, `sampling_share/<situation>` (where SGS puts the
episodes), contacts and penalties per boat.

## Known limits

- Steady (oscillating) wind only; the situation sampler refuses `-weather` presets.
- `t_solo` comes from the hand-written sailor, so "double the projected time" is double *its* time; where it
  sails a setup badly the block is easy to escape.
- The two outcome types are distinct only because of the deadline: without it, "stop the opponent getting there
  first" is the same game as "get there first".
- Evaluating against `digital` in block situations measures little: it always races and never blocks.
