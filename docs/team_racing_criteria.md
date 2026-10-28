# Team racing: what counts as "team tactics emerged"

Written 2026-10-27 ~11:55 UTC, while T2 was still training and before any T2 evaluation was run. The point is to
fix the test before seeing the result, so the GIFs cannot be read to fit it afterwards.

## The comparison

Three agents, all from F3 (`runs/train/f3_aggressive_20261027-100113`), same boats, physics, weather, rules,
opponent pool and PPO settings; only the score and the amount of training differ:

| Agent | Training after F3 | Score |
|---|---|---|
| F3 | none | own place |
| **F3c** (control) | +300M steps | own place |
| **T2** | +300M steps | 2v2 team result (lower sum of places wins, RRS Appendix D) |

T2 vs F3c is the test: same training, different objective. F3 shows whether 300M more steps alone change anything.

All evaluation runs with the training conditions: `--env race-weather --set rrs16=true current_max_frac=0.06
shadow_min_factor=0.6 shadow_length_bl=8.0`, fixed race seeds, every seed of each run.

## Primary test: does the team objective win team races?

`tournament.py --teams`, 2v2, each race one agent's team against another's, 1024 races.

- **H1:** T2 teams beat F3c teams in more than 50% of team races, with the 95% interval above 50%.
- **Check H1b:** in the individual 4-boat tournament (no teams), T2 is **not** better than F3c. If T2 is also
  better individually, H1 could be general racing skill rather than team play, and H1 alone would not show
  team tactics.

## Behaviour: do the boats do what team racers do?

`team_tactics.py`, every boat sailed by the same agent, 2v2 scoring, 1024 races, same races for each agent.
Each measure compares T2 with F3c; "higher" means the 95% intervals do not overlap.

- **H2a, pass-backs:** the team's leader at mark 1 finishes behind its teammate. T2 higher than F3c. In
  individual racing this only happens by chance; in team racing the leader drops back to slow opponents.
- **H2b, shadow on opponents, not the teammate:** per boat, seconds in one opponent's shadow over seconds in the
  teammate's (1 = no preference). T2 at least 25% higher than F3c.
- **H2c, the leader sails for the team:** the race winner's finish time is slower in T2 than in F3c (the leader
  spends time controlling opponents instead of sailing the fastest race).

Reported but with no prediction: the winning combinations (1-2, 1-3, 2-3). Both teams in a self-race use the
same tactics, so the direction is not clear in advance.

## Seeds

Each run has 3 seeds; seed i of T2 and F3c both start from seed i of F3. H1 is tested per pair (T2 seed i vs F3c
seed i, 1024 team races each); H1, H1b and H2a–c each hold if they hold for at least 2 of the 3 seed pairs.

## Verdict

- **Team tactics emerged:** H1 holds and at least two of H2a–c hold.
- **Partial:** only H1, or only the behaviour measures. Report what moved.
- **Did not emerge:** neither.

If H1b fails (T2 better individually too), the result is reported as "the team objective trained a better
racer", not as team tactics, unless the behaviour measures also hold.

## Result (added 2026-10-27 ~13:10 UTC, after the evaluation)

Runs: T2 `runs/train/t2_team_20261027-112002`, F3c `runs/train/f3c_control_20261027-120017`. Outputs in
`runs/tournament/t2_vs_f3c_s{0,1,2}`, `t2_teams`, `t2_individual`, `runs/team_tactics/t2`, `runs/fleet_tactics/t2`.

| | seed 0 | seed 1 | seed 2 | Holds? |
|---|---|---|---|---|
| H1: T2 team wins vs F3c (95% CI) | 37% [34, 40] | 35% [32, 38] | 31% [28, 34] | **no** (0/3, T2 loses) |
| H1b: individual place T2 / F3c | 2.81 / 1.97 | 2.90 / 1.95 | 2.89 / 2.12 | yes (T2 is worse) |
| H2a: pass-backs per race T2 / F3c | 0.61 / 0.21 | 0.54 / 0.15 | 0.62 / 0.13 | yes (3/3) |
| H2b: shadow ratio T2 / F3c (need +25%) | 1.01 / 1.01 | 1.03 / 1.03 | 1.10 / 0.99 | **no** (0/3) |
| H2c: winner's time T2 / F3c (s) | 729 / 682 | 722 / 683 | 717 / 686 | yes (3/3) |

**Verdict by the rules above: partial (behaviour measures only, H2a and H2c).** Read with the rest of the
evidence, it is weaker than that: T2 is a worse racer on every measure (solo time trial 731-755 s vs F3c 704-706
s; individual place 2.8-2.9 vs 2.0-2.1; penalties 0.77-0.82 vs 0.62-0.65 per race), and a slower, sloppier fleet
also produces more pass-backs and slower winners. No team tactics shown; team scoring made the racer worse. F3c
(the control) is slightly better than F3 (team races 61-65% vs F3's 55%), so extra training alone helps a little.

Unexplained: in T2's own races the winner's teammate finishes last in 65-70% of races (2-3 wins; F3c 32-34%,
about chance). To investigate before the next training run.

## Watched in the GIFs, not counted

Named team-racing moves to look for, only to describe what happened, never to decide the verdict: a leader
slowing an opponent on the beat (sitting on its wind) or at a mark (mark trap); a boat luffing or blocking
downwind to let its teammate through; a team in 1-4 attacking the opponents' second boat.
