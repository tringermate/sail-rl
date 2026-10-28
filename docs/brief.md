# Sailing Starts RL — Brief for Claude Code

Oct 17, 2026 · @Matecito

## Goal and context

Build a JAX simulator of J/70 fleet-racing starts and train agents on it with PPO and Success-Guided Sampling (SGS). The point is to learn RL end to end and be able to show the work.

- I race J/70s in fleets but I'm not an expert sailor, and I'm learning RL. Learning matters more than speed of delivery.
- Deliverables: a fast JAX start-line environment, a PPO implementation I understand line by line, an SGS curriculum with baselines, multi-seed experiments with confidence intervals, a write-up, and (stretch) a browser demo where a human races the trained fleet.
- Hardware: \[fill in: CPU only, or GPU model\]. Time per week: \[fill in\].
- Inspiration: the SGS paper (Zhang et al., CoRL 2026), which trains on the configurations the policy succeeds on sometimes but not always.

## How to work with me

Work in phases, explain before you build, and stop at each phase gate for my review.

- Start by reading this whole brief. Ask me any clarifying questions, then propose a plan for Phase 1 only and wait for my go-ahead.
- Before writing a non-trivial component, explain in a few sentences what it does and why it is designed that way.
- Prefer readable over clever: plain JAX, short functions, and comments that map code to the math (GAE, clipped objective, SGS weights).
- PPO, pick one: (a) I write the core update myself and you review it, or (b) you write it with heavy comments and then walk me through it. \[fill in\]
- At the end of each phase, run the acceptance checks, show me plots and GIFs, summarise what worked and what didn't, then stop.
- When training fails, diagnose before changing things: say what you think is wrong, which metric shows it, and the one change you will try.
- Every physics or rules function gets a unit test before it is used in training.
- Ask before adding any dependency not listed under Tech stack. Make small commits with clear messages.

## Tech stack

Python 3.11+, JAX with jit, vmap and scan throughout, and Flax + Optax for networks and optimisation.

- Environment API: functional and gymnax-style. `reset(key, params) -> (obs, state)` and `step(key, state, action, params) -> (obs, state, reward, done, info)`, with state as a pytree.
- Multi-agent: stacked arrays with a leading agent axis (better for speed and parameter sharing than per-agent dicts). Borrow naming conventions from JaxMARL where it helps.
- Training loop: PureJaxRL style. The whole rollout and update compile under jit, environments batch with vmap, and seeds batch with vmap.
- Policy distribution: implement the categorical by hand with `jax.nn.log_softmax`; no extra library.
- Plots with matplotlib; animations exported to GIF or MP4.
- Config: dataclasses (or a small YAML loader), one config per experiment, saved next to its results.
- Packaging with uv and a `pyproject.toml`; tests with pytest.
- Must run on CPU for development and use a GPU when one is present.

## Simulator spec

A 2D kinematic model driven by a polar table is enough; no fluid dynamics. Every number below is a starting value to tune, kept in config.

### World

- Units: metres, seconds and radians internally; knots only at the edges (1 kn = 0.5144 m/s).
- True wind from a base direction (default north) at a true wind speed (TWS) of 6–16 kn. Optional oscillating shift with amplitude, period and phase per episode.
- Start line between a committee boat (starboard end) and a pin. Length = 1.25 × fleet size × boat length, minimum 3 boat lengths. Line bias from −15° to +15° off square to the wind.
- Boat: J/70, length about 6.9 m, beam about 2.25 m.
- One decision per second; physics sub-steps of 0.25 s.

### Boat model

- State per boat: x, y, heading, speed, tack, manoeuvre in progress, started flag, OCS flag, penalty state.
- Target speed = polar(TWA, TWS), bilinear interpolation over `data/polar_j70_approx.csv`. Create a plausible placeholder polar for a 7 m sportboat and label it clearly as approximate; I'll replace it with real data later.
- Speed relaxes toward target speed with separate acceleration and deceleration time constants (start with about 4 s and 8 s). Without this lag the task is trivial.
- No-go zone: |TWA| below about 35° gives target speed 0, so the boat coasts down.
- Luff action: sails eased, target speed about 10–15% of polar. This is how boats hold position near the line.
- Turn rate is limited (about 20°/s at speed, less when slow, no steerage below about 0.5 kn). A tack or gybe swings the heading over a few seconds and costs extra speed (start with 30% for a tack).

### Actions (v1, discrete)

1. Hold course
2. Head up 5°
3. Bear away 5°
4. Head up 15°
5. Bear away 15°
6. Tack or gybe
7. Luff (ease sails for this step)

### Timing, line and OCS

- Solo episodes run from T−60 s to T+10 s (70 steps); fleet episodes from T−90 s to T+10 s.
- A start is a crossing of the line segment from the pre-start side to the course side, detected by segment intersection on the bow's path.
- At T=0, a boat on the course side is OCS. It must return to the pre-start side and cross again (a simplified individual recall).
- Penalty flags (Phase 5 only): none; I flag (over in the last minute means returning around an end); Z flag (fixed scoring penalty); U and black flag (over in the last minute means disqualified; treat both the same in the sim).

### Success (boolean, used by SGS and evaluation)

- Solo: crossed legally between T=0 and T+3 s at 90% or more of polar target speed.
- Fleet: at T+10 s the boat has started legally, has no uncleared foul, is at 90% or more of target speed, and is in clear air (outside every other boat's wind-shadow cone).

### Reward

- Default: sparse terminal reward, +1 for success and −1 for an uncleared OCS or a foul where the boat is at fault.
- Shaping is an experiment, not a default. If added, make it potential-based (e.g. on time-to-line versus time-to-gun) so the optimal policy is unchanged, and compare against sparse.
- Fleet: a relative term (rank by distance gained upwind along the wind axis at T+10 s) is a separate experiment from the boolean success.

### Fleet additions (Phase 4)

- 8–16 boats sharing one policy.
- Observation per boat, in its own frame: own speed, TWA, tack, time to gun, vectors to both line ends, signed distance to the line, line bias. Plus the 5 nearest boats: relative position, relative heading, speed, tack, a flag for whether I have right of way over them, and a flag for whether I'm in their wind shadow. Zero-pad when there are fewer boats.
- Wind shadow: a cone downwind of each boat, about 6 boat lengths long with a half-angle of about 12°. The wind-speed multiplier rises from about 0.75 at the boat to 1.0 at the tip. Tunable.
- Right of way, as one pure function `who_must_keep_clear(a, b)`: port keeps clear of starboard; on the same tack and overlapped, windward keeps clear; on the same tack and not overlapped, the boat clear astern keeps clear; a boat that is tacking keeps clear.
- Contact: boats are capsules of length × beam. On contact, the keep-clear boat takes a two-turns penalty, modelled as a fixed time loss. If no rule assigns blame, both are penalised.
- Unit tests for canonical scenarios: port/starboard crossing, leeward boat luffing a windward boat, a boat tacking into another's path, and a boat clear astern running into one clear ahead.

## Curriculum: Success-Guided Sampling

SGS changes only which task configuration an environment resets to. It favours configurations the policy succeeds on sometimes, but not always.

- A task configuration is one pre-sampled record: initial position, heading and speed of every boat; line bias; TWS; wind-shift amplitude, period and phase; fleet size (Phase 4+); penalty flag (Phase 5).
- Before training, sample a fixed set of N configurations. Choose N so each one is revisited often enough to fill its history: start with N = 4,096 in the solo phase and scale with the number of parallel environments.
- Each configuration keeps a circular buffer of its last H boolean outcomes (default H = 100). Its success rate p is the buffer mean. An empty buffer counts as p = 0, so sampling starts uniform, as in the paper.
- Weights and sampling, from the paper's Eq. 1–2 and Appendix B.2:

```latex
w_i = (p_i + \epsilon)^{\kappa t}\,(1 - p_i + \epsilon)^{\kappa (1 - t)}, \qquad \ell_i = \log\big(\max(w_i, \epsilon) + \epsilon\big), \qquad P(i) = \operatorname{softmax}_i\big(\ell_i / \max(T, 1)\big)
```

- Defaults: target t = 0.5, concentration κ = 1, temperature T = 2, ε = 1e-4. Also try t = 0.66 and κ = 5 (the paper's locomotion setting).
- API, pure JAX and part of the training state: `init(N, H)`, `update(state, config_ids, successes, mask)`, `probs(state)`, `sample(state, key, n)`. An auto-reset wrapper takes a config id, so a finished episode resets to a freshly sampled configuration.
- Fleet: push every boat's outcome into its configuration's buffer, so p is the fraction of boat-starts that succeeded. Flag it to me if this looks wrong in practice.
- Baselines: uniform sampling, and Sampling for Learnability (weight proportional to p(1 − p)). PLR only if time allows.
- Log the SGS weight of a handful of fixed configurations over training, to reproduce the paper's per-configuration plot.
- The configuration set must include too-hard cases (e.g. far from the line at T−60 s) and trivial ones (e.g. on the line at full speed at T−5 s). In the paper, SGS only beat uniform sampling on hard tasks.

## PPO

One readable file, `algos/ppo.py`, in the style of PureJaxRL: actor-critic with GAE and the clipped objective, all compiled under jit.

- Network: separate actor and critic MLPs, two hidden layers of 64 (or 128) with tanh, orthogonal initialisation.
- Loop: a `lax.scan` rollout over vmapped environments, then GAE, then several epochs of shuffled minibatch updates, then logging.
- Starting hyperparameters: learning rate 3e-4 with linear decay, γ = 0.99, λ = 0.95, clip 0.2, entropy coefficient 0.01, value coefficient 0.5, 4 epochs, 4 minibatches, gradient-norm clip 0.5.
- Log every update: mean return, success rate, episode length, entropy, approximate KL, clip fraction, explained variance of the value function.
- Seeds: vmap over seeds so one run gives 5 or more; report mean and 95% confidence intervals.
- Sanity check before any sailing: solve gymnax CartPole-v1 (return about 500). Don't move on until it does.
- Fleet: the same PPO with parameter sharing; each boat is just another batch entry.

## Phases and acceptance criteria

Each phase ends at a gate: the checks below pass, you show me the results, and you wait for my go-ahead.

1. **Phase 1, solo environment.** One boat, polar, speed model, line, OCS, renderer.
   - Unit tests pass for polar interpolation, speed lag, no-go zone, tack loss, line crossing and OCS.
   - A scripted heuristic (sail away, turn back, time the run) produces a plausible GIF.
   - `step` works under jit and vmap, with a printed throughput benchmark at 1K and 4K environments.
2. **Phase 2, PPO and solo timing.** PPO from scratch, then the solo timing task with uniform sampling.
   - CartPole-v1 is solved.
   - Success-rate curve over 5 seeds with 95% CI, plus a GIF of the trained boat.
   - Sparse versus shaped reward, compared on the same seeds.
3. **Phase 3, SGS.** Add the sampler and the baselines.
   - Uniform versus SFL versus SGS over 5+ seeds at 2–3 environment counts (e.g. 512, 4K, and 32K if hardware allows), with 95% CIs.
   - SGS weight against training iteration for a handful of fixed configurations.
4. **Phase 4, fleet.** Multiple boats, wind shadow, the right-of-way subset, contact penalties, self-play with one shared policy.
   - Rule unit tests pass on the canonical scenarios.
   - GIF of a trained fleet start.
   - Metrics: legal-start rate, contacts per start, and how boats spread along the line as bias changes.
5. **Phase 5, headline experiment: penalty flags.** Add the flag to the configuration.
   - For none, I, Z and U/black: OCS rate, mean distance to the line at the gun, success rate. One clear chart.
6. **Phase 6 (stretch), browser demo.** Rewrite the step function in JavaScript, export policy weights to JSON, steer one boat with the keyboard against the trained fleet.
   - Parity test: JavaScript and JAX trajectories match within a small tolerance on recorded episodes.
   - Static site deployable to GitHub Pages.
7. **Phase 7, write-up.** README or blog post: motivation, environment design, what failed and why, results, one-command reproduction.

## Repo layout

A starting layout; adjust it if something reads better, but tell me why.

```text
sailstart/
  env/          polar.py, wind.py, physics.py, rules.py, start_env.py, fleet_env.py, render.py
  curriculum/   sgs.py, uniform.py, sfl.py
  algos/        ppo.py
  experiments/  configs/*.yaml, run_*.py, analysis notebooks
  data/         polar_j70_approx.csv
  tests/
  demo/         Phase 6 web page and exported weights
  docs/         write-up and figures
pyproject.toml
README.md
```

## Out of scope for v1

- Racing rules beyond the subset above: room, proper course, mark-room, and the rules on acquiring right of way and changing course.
- Waves, heel, continuous sail trim, crew weight, and any fluid dynamics.
- Racing beyond T+10 s and marks other than the line. Current may come later as a uniform drift.
- Real-time multiplayer, and opponents learned from human data.

## References

- Zhang et al., [A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control](https://arxiv.org/abs/2610.12465), CoRL 2026: SGS method and hyperparameters. [Project page](https://sgs-rl.github.io/).
- Rutherford et al., "No Regrets: Investigating and Improving Regret Approximations for Curriculum Discovery", NeurIPS 2024: Sampling for Learnability.
- Jiang et al., "Prioritized Level Replay", ICML 2021.
- [PureJaxRL](https://github.com/luchris429/purejaxrl): reference single-file PPO in JAX.
- [gymnax](https://github.com/RobertTLange/gymnax): functional environment API.
- [JaxMARL](https://github.com/bold-lab-ai/JaxMARL): multi-agent conventions in JAX.
- [Agent based match racing simulations: Starting practice](https://research.chalmers.se/en/publication/531213), Chalmers 2022: prior work on starts using Monte-Carlo tree search.
