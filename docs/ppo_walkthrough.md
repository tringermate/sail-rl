# PPO walkthrough

A guided reading of [`sail_env/train/ppo.py`](https://github.com/tringermate/sail-env/blob/main/sail_env/train/ppo.py). Each section explains
one part of the algorithm, points to the lines that implement it, and ties it to something we
actually saw in our sailing runs. Read it with the code open next to it.

## The big picture

The boat's behaviour is a **policy**: a small neural network (the *actor*) that looks at the
observation (speed, wind angle, time to gun, where the line ends are, ...) and outputs a
probability for each of the 7 actions. Training repeats one loop, called an **update**:

1. **Rollout:** let the current policy sail for a while in 512 parallel simulations and record
   everything (observation, action, reward, ...).
2. **Advantages:** for every decision, estimate "was this action better or worse than what I
   normally get from this situation?"
3. **Update:** nudge the policy so better-than-expected actions become a bit more likely and
   worse ones a bit less likely, without changing it too much in one go.
4. **Metrics:** record diagnostics so we can see what is happening.

One update in our sailing runs is 512 envs × 32 steps = 16,384 decisions. A 10M-step run is about
610 updates. All of it runs compiled (`jax.jit`), and five independent seeds train at once
(`jax.vmap`), which is why we get five curves and confidence intervals from one run.

## 1. Two networks: actor and critic (lines 109–136)

- **Actor** ([line 116](https://github.com/tringermate/sail-env/blob/main/sail_env/train/ppo.py#L116)): observation → 7 *logits*, one score per
  action. Softmax turns scores into probabilities.
- **Critic** ([line 128](https://github.com/tringermate/sail-env/blob/main/sail_env/train/ppo.py#L128)): observation → one number, V(s), its
  prediction of the total future reward from this situation ("how good is it to be here?").

Both are two layers of 64 units with tanh. The initialisation matters more than it looks: the
actor's last layer starts with tiny weights (gain 0.01), so the first policy is almost exactly
uniform, 1/7 per action. That is why entropy starts at 1.946 = ln 7 in every run: the agent begins
maximally exploratory.

## 2. The action distribution (lines 140–159)

We wrote the categorical distribution by hand, as the brief asks:

- `categorical_sample` picks an action according to the probabilities.
- `categorical_log_prob` is log π(a|s), the log-probability of the action actually taken. PPO
  compares this number before and after an update.
- `categorical_entropy` measures how random the policy still is. High entropy = still exploring;
  near 0 = always doing the same thing.

**In our runs:** entropy fell from 1.95 to about 1.0–1.2 in the sailing runs that learned. The
policy became decisive but not deterministic. In the parking runs, entropy also fell; a policy
can become confident about the wrong thing.

## 3. Rollout (lines 399–444)

`env_step` ([line 399](https://github.com/tringermate/sail-env/blob/main/sail_env/train/ppo.py#L399)) does one step in all 512 simulations: run
the actor, sample actions, run the critic (we need V(s) later), step the environments. `lax.scan`
repeats it 32 times ([line 444](https://github.com/tringermate/sail-env/blob/main/sail_env/train/ppo.py#L444)), giving a block of transitions of
shape (32 steps, 512 envs).

The environments reset themselves when an episode ends, so a block of 32 steps contains the ends
of some episodes and the beginnings of others. That is why we needed **staggering**: without it
all 512 boats finished on the same step, and two out of three blocks contained no finished start
at all, so no reward.

A few lines before it ([line 436](https://github.com/tringermate/sail-env/blob/main/sail_env/train/ppo.py#L436)) is where the **staged rules**
plug in: the environment's rules are recomputed from training progress every update.

## 4. Advantages with GAE (lines 191–220)

This is the heart of "which actions were good". For each step:

- **TD error:** δₜ = rₜ + γ·V(sₜ₊₁) − V(sₜ). What happened (reward now, plus the critic's estimate
  of where we ended up) minus what the critic expected. Positive = better than expected.
- **GAE:** Aₜ = δₜ + γλ·Aₜ₊₁, computed backwards in time. It blends the one-step surprise with
  later surprises, discounted by γλ = 0.99 × 0.95. λ trades off bias (λ = 0 trusts the critic
  fully) against noise (λ = 1 waits for the actual outcome).
- (1 − dₜ) cuts the chain at the end of an episode, so the next episode's values don't leak in.

The **return target** Rₜ = Aₜ + V(sₜ) is what the critic is trained towards.

This is tested against a plain double loop and against the λ = 1 case (discounted return) in
[`tests/test_ppo.py`](https://github.com/tringermate/sail-env/blob/main/tests/test_ppo.py).

**In our runs:** with a sparse reward, a successful start pays +1 only at T+10. GAE is what carries
that +1 back to the decisions 30 or 50 seconds earlier (the tack onto starboard, the timing of the
run). With γ = 0.99, a reward 50 steps later still counts for 0.99⁵⁰ ≈ 0.6.

## 5. The PPO loss (lines 224–286)

For each minibatch, three terms:

**a) Policy loss, the clipped surrogate** ([lines 244–257](https://github.com/tringermate/sail-env/blob/main/sail_env/train/ppo.py#L244-L257)):

- ratio r = π_new(a|s) / π_old(a|s): how much more (or less) likely the action is now than when it
  was taken.
- objective = min( r·A, clip(r, 0.8, 1.2)·A ).

If an action was good (A > 0), increasing its probability helps, but only until it is 20% more
likely; beyond that the clip makes the gradient zero. Same for bad actions in the other direction.
This is the "proximal" in PPO: each update can only move the policy a little, so one lucky or
unlucky batch cannot wreck it.

*Decision point:* advantage normalisation ([lines 250–255](https://github.com/tringermate/sail-env/blob/main/sail_env/train/ppo.py#L250-L255)) rescales A
to mean 0, std 1 in each minibatch. It makes the step size independent of the reward's scale.
We kept it on.

**b) Value loss** ([lines 259–264](https://github.com/tringermate/sail-env/blob/main/sail_env/train/ppo.py#L259-L264)): squared error between V(s)
and the return target Rₜ. *Decision point:* value clipping limits how far V can move per update,
like the policy clip. On (PureJaxRL default); research suggests it rarely matters.

**c) Entropy bonus** ([lines 266–267](https://github.com/tringermate/sail-env/blob/main/sail_env/train/ppo.py#L266-L267)): a small reward (coefficient
0.01) for staying random, to keep exploring.

Total = policy loss + 0.5 × value loss − 0.01 × entropy.

## 6. The update loop (lines 450–480)

The 16,384 transitions are reused for **4 epochs**; each epoch shuffles them and cuts them into
**4 minibatches** of 4,096, with one gradient step each, so 16 gradient steps per update. Gradients
are clipped to norm 0.5 and the learning rate decays linearly to 0 over training
([lines 342–343](https://github.com/tringermate/sail-env/blob/main/sail_env/train/ppo.py#L342-L343)).

## 7. Reading the diagnostics (lines 482–562)

| Metric | Meaning | What we saw |
|---|---|---|
| `approx_kl` | how much the policy changed in this update | tiny (≤ 0.001): updates were gentle, and fell to 0 as the learning rate decayed |
| `clip_frac` | share of samples that hit the 0.8/1.2 clip | near 0: the clip rarely bit; updates were small anyway |
| `entropy` | how random the policy is | 1.95 → ~1.1 in successful runs |
| `explained_variance` | how well the critic predicts returns (1 = perfect) | 0.97 in the shaped runs, which is the clue below |
| `value_loss` | the critic's error | collapsed to ~0 in the parking runs: "predict 0 everywhere" was trivially right |

## 8. The lessons from our runs, in PPO terms

**Why the shaped reward didn't help.** Potential-based shaping adds γΦ(s′) − Φ(s) to each reward.
The critic learns V_shaped(s) = V(s) − Φ(s), and then the TD errors, and so the advantages, are
exactly the same as without shaping: the Φ terms cancel. Explained variance of 0.97 meant the
critic had learned Φ almost perfectly within about a million steps, after which the shaping was
mathematically invisible to the policy. Shaping can only help while the critic is still wrong.

**Why the sparse runs collapsed into parking.** Advantages are relative to V(s). Early on, the
critic sees OCS (−1) far more often than success (+1), so actions near the line look worse than
expected and actions that stay away look slightly better. PPO faithfully followed those
advantages, and once nobody reached the line, there were no successes left to learn from.

**Why the curriculum worked.** PPO can only reinforce what it experiences. Easier starts made
success common enough to show up in the advantages; the staged rules (no OCS penalty, a 10 s
window, any speed) made the first successes cheap, and tightening them gradually kept the policy
succeeding while the target moved. In run C the OCS rate peaked at 55% and then fell as the
penalty phased in.

**Why the starboard approach bonus was gamed.** A per-step bonus is not potential-based, so it
changes what is optimal. Sitting on starboard luffing earned up to 0.3 with certainty; a real start
was risky. For two of five seeds, the advantages said "parking on starboard is better than
expected", and PPO did exactly that. Fading the bonus out (run E2) removes that reward by the end.

## Glossary

- **Episode:** one start sequence, T−60 to T+10.
- **Return:** total (discounted) reward from a point to the end of the episode.
- **Policy π(a|s):** probability of action a in situation s (the actor).
- **Value V(s):** expected return from s (the critic).
- **Advantage A:** how much better an action turned out than V(s) expected.
- **γ (gamma):** discount, 0.99: a reward k steps later counts 0.99ᵏ.
- **λ (lambda):** GAE's bias/noise trade-off, 0.95.
- **Seed:** an independent training run with different random numbers; five seeds give the
  confidence intervals.
