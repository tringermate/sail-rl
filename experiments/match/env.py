"""MatchEnv: 1v1 episodes from the situation bank, with roles, a decided outcome and a zero-sum reward.

Every episode resets to one bank entry (a setup x a role pairing, chosen by the curriculum sampler) or, with
probability full_race_frac, to a full race just after the start. The two boats are sailed by the same policy;
the observation tells each boat its own and the opponent's role.

Outcomes (t_solo = the boat's undisturbed time to the target, situations.py):
    race vs race      the first boat to reach the target wins. Nobody there after race_factor x the longer
                      t_solo (or at the race's time limit): the boat with less distance to sail wins (a tie
                      within tie_margin m).
    block vs escape   the escaper wins if it reaches the target by its deadline, else the blocker wins. The
                      deadline is block_factor x t_solo after the setup; for a start, block_factor x the time
                      after the gun (the time before the gun cannot be sailed faster):
                          deadline = g + block_factor x (t_solo - g),   g = seconds to the gun at the setup (>= 0)
The blocker's own progress does not count.

Reward per boat: +outcome_reward win / -outcome_reward loss / 0 tie at the end; sail_env's contact costs
(contact_cost, contact_cost_row) at every contact; -foul_cost per penalty incurred (so a deliberate foul to
block does not pay); optional potential-based shaping (Phi = -own distance to sail / 100 m for race and escape,
+opponent's distance to sail / 100 m for block; Phi = 0 at the end, so it cannot change the best policy).

Observation: the RaceEnv observation, then MATCH_FEATURES: own role one-hot (race, block, escape), the
opponent's role one-hot, time left until the horizon or deadline / 100 s (at most 6), 1 if the target is the
start checkpoint.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from experiments.match.situations import (BLOCK, ESCAPE, FULL_RACE, KIND_FINISH, KIND_START, NAMES, RACE, Bank,
                                          target_reached)
from sail_env.fleet_rules import shadow_factors
from sail_env.physics import bow_position
from sail_env.race_env import N_LEGS, rounding_to_go
from sail_env.train.curriculum import sampler as smp
from sail_env.train.curriculum.config_env import Curriculum

MATCH_FEATURES = 8


@struct.dataclass
class MatchParams:
    env: object  # EnvParams of the 2-boat race
    cdf: jax.Array  # (n_sit,) cumulative sampling probabilities over the situation entries (the curriculum's)
    block_factor: float = 2.0
    race_factor: float = 2.0
    outcome_reward: float = 1.0
    foul_cost: float = 0.5
    shaping: float = 0.0
    full_race_frac: float = 0.0
    tie_margin: float = 2.0  # m


@struct.dataclass
class MatchState:
    fs: object  # FleetState
    cid: jax.Array  # config id (situation entry), -1 for a full race
    t0: jax.Array  # clock at the setup
    leg0: jax.Array  # (2,)
    kind: jax.Array
    role: jax.Array  # (2,)
    t_solo: jax.Array  # (2,)
    reach_t: jax.Array  # (2,) s after the setup when the target was reached (inf until then)
    phi: jax.Array  # (2,) shaping potential of the current state


class MatchEnv:
    def __init__(self, race_env, bank: Bank, n_sit: int, start_checkpoint: float):
        self.env = race_env
        self.bank = bank
        self.n_sit = n_sit
        self.n_full = int(bank.kind.shape[0]) - n_sit
        self.start_checkpoint = start_checkpoint
        self.num_agents = race_env.num_agents
        assert self.num_agents == 2, "match racing: fleet_size must be 2"
        self.num_actions = race_env.num_actions
        self.obs_dim = race_env.obs_dim + MATCH_FEATURES

    # --- outcome helpers -------------------------------------------------------------------------------------

    def to_go(self, fs, params):
        """(2,) m still to sail by the rounding route (0 once finished)."""
        k = self.env.course(fs, params)
        bx, by = bow_position(fs.boats.boat, params)
        d = rounding_to_go(fs.leg, bx, by, k, 2.0 * params.boat_length, N_LEGS, fs.gate_through, fs.mark_through)
        return jnp.where(fs.leg >= N_LEGS, 0.0, d)

    def deadline(self, s: MatchState, params: MatchParams):
        """(2,) s after the setup: each boat's escape deadline (block_factor x its undisturbed time after the gun)."""
        g = jnp.maximum(-s.t0, 0.0)
        return g + params.block_factor * (s.t_solo - g)

    def horizon(self, s: MatchState, params: MatchParams):
        """s after the setup when the episode ends at the latest (a full race: at the race's time limit)."""
        race = jnp.all(s.role == RACE)
        esc = jnp.min(jnp.where(s.role == ESCAPE, self.deadline(s, params), jnp.inf))
        h = jnp.where(race, params.race_factor * jnp.max(s.t_solo), esc)
        return jnp.where(s.kind == KIND_FINISH, jnp.inf, h)

    def phi(self, s: MatchState, fs, params):
        d = self.to_go(fs, params.env)
        return jnp.where(s.role == BLOCK, d[::-1], -d) / 100.0

    # --- API -------------------------------------------------------------------------------------------------

    def entry(self, idx, cid, params: MatchParams):
        b = jax.tree.map(lambda x: x[idx], self.bank)
        s = MatchState(fs=b.fs, cid=cid, t0=b.fs.boats.t[0], leg0=b.leg0, kind=b.kind, role=b.role, t_solo=b.t_solo,
                       reach_t=jnp.full(2, jnp.inf), phi=jnp.zeros(2))
        s = s.replace(phi=self.phi(s, b.fs, params))
        return self.get_obs(s, params), s

    def reset(self, key, params: MatchParams):
        k_cfg, k_full, k_idx = jax.random.split(key, 3)
        cid = smp.sample(params.cdf, k_cfg)
        if self.n_full:  # static
            full = jax.random.uniform(k_full) < params.full_race_frac
            idx = jnp.where(full, self.n_sit + jax.random.randint(k_idx, (), 0, self.n_full), cid)
            cid = jnp.where(full, -1, cid)
        else:
            idx = cid
        return self.entry(idx, cid, params)

    def step_env(self, key, s: MatchState, action, params: MatchParams):
        ep = params.env
        _, fs, r_env, done_env, _ = self.env.step_env(key, s.fs, action, ep)
        el = fs.boats.t[0] - s.t0
        hit = target_reached(self.env, fs, s.kind, s.leg0, self.start_checkpoint, ep)
        reach_t = jnp.where(jnp.isinf(s.reach_t) & hit, el, s.reach_t)
        reached = jnp.isfinite(reach_t)
        horizon = self.horizon(s, params)
        timeout = (el >= horizon) | done_env[0]
        to_go = self.to_go(fs, ep)

        # race vs race: first there; else less distance to sail
        first = jnp.where(reach_t < reach_t[::-1], 1.0, jnp.where(reach_t > reach_t[::-1], -1.0, 0.0))
        ahead = jnp.where(to_go < to_go[::-1] - params.tie_margin, 1.0,
                          jnp.where(to_go > to_go[::-1] + params.tie_margin, -1.0, 0.0))
        race_res = jnp.where(jnp.any(reached), first, ahead)
        race_done = jnp.any(reached) | timeout
        # block vs escape
        esc = s.role == ESCAPE
        esc_t = jnp.min(jnp.where(esc, reach_t, jnp.inf))
        esc_deadline = jnp.min(jnp.where(esc, self.deadline(s, params), jnp.inf))
        escaped = esc_t <= esc_deadline
        block_res = jnp.where(esc == escaped, 1.0, -1.0)
        block_done = escaped | timeout

        racing = jnp.all(s.role == RACE)
        done = jnp.where(racing, race_done, block_done)
        result = jnp.where(done, jnp.where(racing, race_res, block_res), 0.0)

        s2 = s.replace(fs=fs, reach_t=reach_t)
        phi2 = jnp.where(done, 0.0, self.phi(s2, fs, params))
        new_fouls = (fs.fouls - s.fs.fouls).astype(jnp.float32)
        reward = (r_env + params.outcome_reward * result - params.foul_cost * new_fouls
                  + params.shaping * (phi2 - s.phi))
        s2 = s2.replace(phi=phi2)

        wd, _ = self.env.local_winds(fs.boats, ep)
        clear = shadow_factors(fs.boats.boat, wd, ep).min(axis=1) >= 1.0
        dn = jnp.broadcast_to(done, (2,))
        info = {
            "config_id": jnp.broadcast_to(s.cid, (2,)),
            "situation": jnp.broadcast_to(jnp.where(s.cid < 0, FULL_RACE, self.bank.situation[jnp.maximum(s.cid, 0)]), (2,)),
            "role": s.role,
            "result": result,
            "win": result > 0.0,
            "focal": jnp.arange(2) == 0,
            "fouls": jnp.where(dn, fs.fouls, 0),
            "unserved_foul": dn & (fs.penalty_left > 0.0),
            "clear_air": dn & clear,
            "contacts": jnp.where(dn, fs.contacts, 0),
            "mark_hits": jnp.where(dn, fs.mark_hits, 0),
        }
        return self.get_obs(s2, params), s2, reward.astype(jnp.float32), dn, info

    def step(self, key, s: MatchState, action, params: MatchParams):
        """step_env + auto-reset to a freshly sampled entry when the episode ends."""
        k_step, k_reset = jax.random.split(key)
        obs, s2, reward, done, info = self.step_env(k_step, s, action, params)
        obs_re, s_re = self.reset(k_reset, params)
        end = done[0]
        s2 = jax.tree.map(lambda a, b: jnp.where(end, a, b), s_re, s2)
        obs = jnp.where(end, obs_re, obs)
        return obs, s2, reward, done, info

    def get_obs(self, s: MatchState, params: MatchParams):
        base = self.env.get_obs(s.fs, params.env)
        left = self.horizon(s, params) - (s.fs.boats.t[0] - s.t0)
        left = jnp.where(s.kind == KIND_FINISH, 1200.0 - s.fs.boats.t[0], left)  # full race: the race's own limit
        own = jax.nn.one_hot(s.role, 3)
        extra = jnp.concatenate([own, own[::-1], jnp.clip(left / 100.0, 0.0, 6.0)[None].repeat(2)[:, None],
                                 jnp.full((2, 1), (s.kind == KIND_START).astype(jnp.float32))], axis=1)
        return jnp.concatenate([base, extra], axis=1)


class MatchCurriculum(Curriculum):
    """The sampler sees one outcome per episode: boat 0's (the focal boat). In self-play the two boats' outcomes
    always sum to zero, so their mean would say nothing about a setup; boat 0's says how the setup and role
    pairing go for the side it puts boat 0 on (geometry and roles are randomised over boats 0 and 1)."""

    def __init__(self, cfg: smp.SamplerConfig, bank: Bank, n_sit: int):
        super().__init__(cfg, success_key="win")
        self.situation = np.asarray(bank.situation[:n_sit])
        self.role0 = np.asarray(bank.role[:n_sit, 0])

    def make_params(self, params: MatchParams, state):
        return params.replace(cdf=jnp.cumsum(smp.probs(state, self.cfg)))

    def update(self, state, info):
        mask =info["returned_episode"] & info["focal"] & (info["config_id"] >= 0)
        return smp.update(state, jnp.maximum(info["config_id"], 0), info["win"], mask)

    def log(self, state):
        out = super().log(state)
        p = smp.success_rate(state)
        seen = state.count > 0
        prob = smp.probs(state, self.cfg)
        block_all, n_all = 0.0, 0
        for sid, name in enumerate(NAMES):
            in_s = self.situation == sid
            if not in_s.any():
                continue
            # block success: boat 0 blocking and winning, or boat 0 escaping and losing
            as_block = jnp.asarray(in_s & (self.role0 == BLOCK)) & seen
            as_esc = jnp.asarray(in_s & (self.role0 == ESCAPE)) & seen
            n = as_block.sum() + as_esc.sum()
            succ = jnp.sum(jnp.where(as_block, p, 0.0)) + jnp.sum(jnp.where(as_esc, 1.0 - p, 0.0))
            out[f"block_success/{name}"] = jnp.where(n > 0, succ / jnp.maximum(n, 1), jnp.nan)
            out[f"sampling_share/{name}"] = jnp.sum(jnp.where(jnp.asarray(in_s), prob, 0.0))
            block_all, n_all = block_all + succ, n_all + n
        out["block_success"] = jnp.where(n_all > 0, block_all / jnp.maximum(n_all, 1), jnp.nan)
        return out
