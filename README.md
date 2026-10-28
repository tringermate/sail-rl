# sail-rl

**Can reinforcement learning learn to race a sailboat? A research log, from the start line to fleet and team
racing, on the [sail-env](https://github.com/tringermate/sail-env) J/70 simulator.**

<p align="center">
  <img src="docs/figures/fleet/gif_f3.gif" width="380" alt="Four boats trained by self-play racing each other">
</p>
<p align="center"><sub>Four boats sailed by the same self-play policy (F3) race two laps in moving weather.</sub></p>

**[▶ Race the trained agents in your browser](https://tringermate.github.io/sail-env/)**: no install needed.


## TL;DR
I started working on this project to get some hands-on experience with RL and MARL, while secretly hoping that some strange but efficient sailing strategies will emerge that I can use in the real life as well;)

Ongoing project probably with many improvements to be made - feel free to let me know if I badly messed up something:)

The creation of the codebase was supported by Claude Code - but I still felt like I was controlling it, hopefully this stays like this for the rest of human history:)


## The question

Sailboat racing combines things that are hard for learning agents: split-second timing with no instrument to
help (be on the line, at full speed, when the gun fires), a long race in which only the final place counts,
a boat that answers with lag, wind that shifts, and other boats that steal your wind and must obey
right-of-way rules. This project asks, one step at a time:

1. **Starts:** can an agent learn a good start, alone and in a fleet, and which training choices matter?
2. **Racing alone:** can it sail a full two-lap race faster than a hand-written expert, also in moving weather?
3. **Fleet racing:** in a 4-boat fleet, does self-play produce a better racer, and do real tactics appear
   (splitting to take a side, covering, using wind shadow)?
4. **Team racing:** with the same boats and only the score changed to the team result (2v2), do team tactics
   appear (passing a teammate back, holding opponents back)?

## Key results

| Step | Result |
|---|---|
| **Starts** | 89% good starts alone (on the line at the gun, fast, on starboard; 86.5% on sail-env's separate baseline set), beating a controller that searches 40 settings per start with knowledge of the future (78%); 68–90% in a 5-boat fleet, depending on line length and shifts. A curriculum over *what counts as success* mattered far more than one over *which situations to practise*. |
| **Racing alone** | Plain PPO never got round the course. A ladder of shorter courses, kickstarting from the hand-written racer (an imitation term that fades to zero), and scale (2048 races in parallel on one GPU) gave racers 21–28 s per race faster than the hand-written racer, in steady wind and in weather. Part of that margin is the hand-written racer's slow gate rounding. |
| **Fleet racing** | Self-play with a pool of opponents reaches a mean place of 2.06 of 4 against the hand-written racer (3.25) and an earlier fleet agent (3.14), finishing 99.5% of races. The boats overtake (7.3 order changes per race), but they sail in a tight pack and never split to take a side. Measured only against opponents they trained with. |
| **Team racing** | **Negative.** Scored for the team, the same boats got worse: they lose 63–69% of team races to a control trained equally long for their own place, and they are slower alone. No team tactics appeared, by criteria written down before the results ([criteria](docs/team_racing_criteria.md)). |

What I take from it:
- **Reward design dominated.** Almost every failure was the agent doing exactly what the reward said:
  parking in irons, farming per-second bonuses, sailing in a procession because contact cost more than any
  pass was worth.
- **Look at the races, not only the numbers.** Two environment bugs (boats cutting the leeward gate, and
  touching the windward mark without going round it) and the procession were found in the race GIFs while
  every metric looked fine.
- **Scale, not a new method, was the last step past the expert** in racing alone.
- **The methods are standard** (PPO, self-play, opponent pools, kickstarting, curricula). What is new is the
  question: no earlier reinforcement-learning work on learned fleet or team racing was found.

## Current results

The best agent at each step against the hand-written "digital" sailor (time-on-distance at the start; laylines,
VMG angles and proper roundings in the race). Ranges are over 3 training seeds.

| Task | Agent | Measure | Learned agent | Hand-written baseline |
|---|---|---|---|---|
| Start, alone | best solo policy | good starts | **89.2 ± 3.1%** | 20% (78% when allowed 40 tries per start, knowing the future) |
| Start, 5-boat fleet | fleet policy | good starts | **68–90%** (with / without shifts and line bias) | 9% |
| Race alone, steady wind | `rl-racer` | finish time vs the digital racer, 64 paired races | **20.7–24.1 s faster**, in 100% of races | – |
| Race alone, weather | `rl-racer-weather` | same | **25.6–28.3 s faster**, in 91–97% of races | – |
| 4-boat fleet, weather | F3 (self-play + opponent pool) | mean place of 4, 512 mixed races | **2.06** [1.96, 2.17] | 3.25 [3.15, 3.33] |
| 2v2 team racing | T2 (team reward) vs control F3c (own place) | team races won | **31–37%** (T2 loses) | – |

## In motion

| Fleet start: 5 boats, staggered, out of each other's wind shadow | Racing alone in weather (W5b, before the gate-rule fix) |
|---|---|
| ![fleet start](docs/figures/phase4_fleet_v4b.gif) | ![racing in weather](docs/figures/racing/gif_w5b_weather.gif) |
| **F2: the procession.** Contact cost more than any pass was worth, so the boats followed each other | **T2, team racing (2v2):** no team tactics appeared |
| ![procession](docs/figures/fleet/gif_f2_procession.gif) | ![team racing](docs/figures/fleet/gif_t2_team.gif) |

More GIFs, one per training stage, are in the write-ups below.

## Write-ups

| | |
|---|---|
| [docs/writeup_starts.md](docs/writeup_starts.md) | Starts: reward design, staged success criteria, curricula (uniform, SFL, SGS), fleet starts |
| [docs/writeup_racing.md](docs/writeup_racing.md) | Racing alone: the parking problem, the ladder, kickstarting, weather, scale |
| [docs/writeup_fleet_team.md](docs/writeup_fleet_team.md) | Fleet and team racing, the control run, and what the environment got wrong |
| [docs/team_racing_criteria.md](docs/team_racing_criteria.md) | The team-racing test, fixed before the results, and its verdict |
| [docs/ppo_walkthrough.md](docs/ppo_walkthrough.md) | How the PPO code works, explained with these runs |

## Installation

sail-rl contains the experiments; the simulator, PPO and agents come from
[sail-env](https://github.com/tringermate/sail-env), installed from a clone next to this one.

```sh
git clone https://github.com/tringermate/sail-env
git clone https://github.com/tringermate/sail-rl && cd sail-rl
uv sync                     # Python 3.12 environment with sail-env (editable) and its dependencies
uv sync --extra wandb       # optional: Weights & Biases dashboards
```

Everything runs on a CPU; JAX uses a CUDA GPU automatically when there is one (Linux or WSL2). The tests live
in sail-env: `cd ../sail-env && uv run pytest`.

## Reproducing the results

Training runs are configs in [experiments/configs/](experiments/configs/). Each run writes
`runs/train/<name>_<time>/` (config, metrics, parameters, learning curves, GIFs). Most configs log to W&B; add
`--no-wandb` to `sail_env.train.run`, or set `wandb: false` in the config. Runs that continue from an earlier
run name it in `init_from` (and fleet opponents in `opponents`) by its folder name from the original runs,
which are not included: point these at your own run folders.

**Starts**

```sh
uv run python -m sail_env.train.run experiments/configs/p3_staged_uniform.yaml   # best solo policy (89%)
uv run python -m sail_env.train.run experiments/configs/fleet_c4.yaml            # 5-boat fleet with shifts and bias
uv run python experiments/evaluate.py runs/train/<run>                           # good starts on standard starts
uv run python experiments/evaluate_fleet.py runs/train/<run>
```

**Racing alone** (stages 1–3 on a CPU; the full race and weather on a GPU)

```sh
uv run python experiments/ladder.py experiments/configs/ladder/s1_beat100.yaml \
    experiments/configs/ladder/s2_beat400.yaml experiments/configs/ladder/s3_beat_run.yaml
uv run python experiments/queue.py experiments/configs/gpu/gpu_s4k_2048.yaml:runs/train/<stage-3 run>
uv run python experiments/queue.py experiments/configs/gpu/gpu_racer_gate.yaml:latest=gpu_s4k_2048 \
    experiments/configs/gpu/gpu_weather_gate.yaml:latest=gpu_racer_gate
uv run python experiments/diagnose_race.py runs/train/<run> --races 256      # paired against the hand-written racer
```

The two resulting racers are bundled in sail-env as `rl-racer` and `rl-racer-weather`.

**Fleet and team racing** (GPU; about 40 minutes per run on an RTX 4090)

```sh
uv run python experiments/queue.py \
    experiments/configs/fleet/f2_weather_league.yaml:../sail-env/sail_env/pretrained/racer-weather \
    experiments/configs/fleet/f3_aggressive.yaml:latest=f2_weather_league \
    experiments/configs/fleet/t2_team.yaml:latest=f3_aggressive \
    experiments/configs/fleet/f3c_control.yaml:latest=f3_aggressive
```

Evaluation, with the conditions the fleet agents trained in:

```sh
SET="rrs16=true current_max_frac=0.06 shadow_min_factor=0.6 shadow_length_bl=8.0"
uv run python experiments/tournament.py F3=runs/train/<f3>#0 digital=digital --env race-weather --boats 4 --set $SET
uv run python experiments/tournament.py T2=runs/train/<t2>#0 F3c=runs/train/<f3c>#0 --env race-weather --boats 4 --teams --set $SET
uv run python experiments/fleet_tactics.py digital=digital F3=runs/train/<f3>#0 --env race-weather --set $SET
uv run python experiments/team_tactics.py F3c=runs/train/<f3c>#0 T2=runs/train/<t2>#0 --env race-weather --set $SET
```

`experiments/runpod/` has the scripts used to run on a rented GPU (pack, set up, start a queue, copy results
back); see its README.

## Repository layout

```text
experiments/
  configs/            one YAML per training run (environment, PPO, curriculum, opponents)
  *.py                training queues and evaluation: evaluate*, tournament, fleet_tactics, team_tactics,
                      diagnose_race, ladder, queue, compare_runs, failure_map, make_writeup_figures
  runpod/             running on a rented GPU
  weather_failures/   the analysis of unfinished races in strong current
docs/                 write-ups, the original project brief, figures and GIFs
```

## Next steps

- **Environment:** measure how much a side of the course is worth, then make the sides differ more and less
  predictably, so that splitting from the fleet becomes a real gamble with a real payoff.
- **Team racing:** find out why team training pulls teammates apart (in the team agent's own races the
  winner's teammate finishes last two times in three), most likely the credit-assignment problem of a shared
  reward; then a mixed reward (own place plus a team bonus) and strong team opponents.
- **Generalisation:** race the fleet agents against opponents they never met, including people in the browser game.

## References

- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017: PPO, the learner throughout.
- Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), ICLR 2016: GAE.
- Schmitt et al., [Kickstarting Deep Reinforcement Learning](https://arxiv.org/abs/1803.03835), 2018: the fading imitation term used for racing.
- Ng, Harada and Russell, "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping", ICML 1999: potential-based shaping.
- Zhang et al., [A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control](https://arxiv.org/abs/2610.12465), CoRL 2026: Success-Guided Sampling (SGS). [Project page](https://sgs-rl.github.io/).
- Rutherford et al., [No Regrets: Investigating and Improving Regret Approximations for Curriculum Discovery](https://arxiv.org/abs/2408.15099), NeurIPS 2024: Sampling for Learnability (SFL).
- Jiang et al., [Prioritized Level Replay](https://arxiv.org/abs/2010.03934), ICML 2021.


## License

MIT ([LICENSE](LICENSE)).
