## 29/10/2026
Based on SGS: A task configuration is a triple (s0, g, e): the robot’s initial state s0, its goal g, and the environment e, such as the terrain type or the layout of the scene. SGS samples a large, fixed set of task configurations before training.

In 1 v 1 sailing one can define a relatively small subset of situations that cover a relatively large part of any give race. If it is set up properly, could enhance the training and learning progress:

idea: for every sitauation (S) create a number (n) situations with 2 dimensions (S_n_d):
where:
- S is the situation defined by the given rule (i.e.: starboard-portside situation)
- n is a predefined number, that represents the number of different set ups (a given distribution in relative position and angle of boats)
-d is the objective of the task, that is twofold (in a matchrace situation 1 v 1 it is twofold, but later can  be larger): 
        - round the upwind/downwind mark or cross the finish/start first 
        - be an a**hole and keep the opponent from rounding the upwind/downwind mark or crossing the finish/start first regardless of your position to these marks


at first let s do without sgs weighting, than include that as well and compare.

Expectation:
If a sailor masters these skillsets, it will be able to use them effectively on a live situation. 

// on a tangent based on this, one could also define defensive or agressive sailors


//larger scope for real world robotics:
just like sgs, it can enhance training performance.

> ---
> **🔎 CRITIQUE (Claude, 29/10/2026)** — *Sensible: yes.* Self-play rarely samples the decisive encounters (port/stbd, mark room, windward/leeward) densely; a situation library fixes that.
>
> - **Objectives overlap in 1v1.** "Stop the opponent rounding first" = "round first yourself", unless a timeout where *nobody* rounds counts as a defender win. Define the time limit and win condition per task, or d₁ and d₂ are the same task.
> - **Rules must bite first.** The blocking objective will learn to foul or ram unless penalties are in place and sized above any blocking gain (your earlier reward-hacking lesson).
> - **Prior evidence cuts against it.** At starts, a curriculum over *what counts as success* beat one over *which situations to practise*. Expect d to matter more than n; test them separately.
> - **One policy, two objectives** needs d as an observation input (or two policies), else the signals conflict.
> - **Transfer isn't automatic.** Judge on full races vs. a plain self-play control (same compute, ≥3 seeds), not per-situation success; mix in full races during training. Keep a held-out set of setups to catch overfitting to the fixed n.
> ---
