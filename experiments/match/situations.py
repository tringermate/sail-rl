"""1v1 situation library: the encounters that decide most match races, as starting states for training.

Each situation is a distribution over two-boat setups (relative position, headings, speeds, clock) placed on
the ordinary race course of sail_env's RaceEnv (2 boats). A setup also fixes the target both boats sail for:

    kind 0  complete the current leg (round the windward mark, pass and round the gate, or finish)
    kind 1  start: cross the line legally, then reach START_CHECKPOINT m upwind of it (timing + speed + lane)
    kind 2  finish the whole race (the full-race entries, from just after a start)

A target only counts with no penalty left to serve. For every setup the hand-written racer (sail_env's
digital sailor) sails each boat ALONE (the other boat removed) to measure t_solo, the undisturbed time to the
target from the setup; setups the digital sailor cannot complete within SOLO_STEPS s are dropped. Starts are
the exception: t_solo is projected (projected_start_times), because the digital sailor starts late from them.

Conventions as in sail_env: compass headings, wind direction = where it blows FROM, signed TWA > 0 =
starboard tack, u = upwind unit vector of the course, r = to the right looking upwind. L = boat length.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from sail_env.agents import digital_sailor_agent
from sail_env.agents.digital_sailor import beat_angle
from sail_env.agents.race_sailor import run_angle
from sail_env.fleet_rules import contacts
from sail_env.geometry import heading_vector, signed_twa, wrap_angle
from sail_env.physics import HOLD, bow_position
from sail_env.polar import polar_speed
from sail_env.race_env import PARK_X
from sail_env.weather import any_weather
from sail_env.wind import wind_at

KIND_LEG, KIND_START, KIND_FINISH = 0, 1, 2
SOLO_STEPS = 240  # s: longest undisturbed time a setup may need
RACE, BLOCK, ESCAPE = 0, 1, 2  # roles
ROLE_NAMES = ["race", "block", "escape"]
# Role pairings per setup: both race for the mark / boat 0 blocks boat 1 / boat 1 blocks boat 0.
PAIRINGS = [(RACE, RACE), (BLOCK, ESCAPE), (ESCAPE, BLOCK)]


@struct.dataclass
class Bank:
    """N training entries (leading axis): a race state plus what the episode is about."""
    fs: object  # FleetState of the 2-boat RaceEnv
    role: jax.Array  # (N, 2) int: RACE / BLOCK / ESCAPE
    kind: jax.Array  # (N,) int: KIND_*
    t_solo: jax.Array  # (N, 2) s: undisturbed time to the target from the setup (inf for full races)
    situation: jax.Array  # (N,) int: index into SITUATIONS (len(SITUATIONS) = full race)
    leg0: jax.Array  # (N, 2) int: leg at the setup


def deg(x):
    return jnp.radians(x)


def _u(key, lo, hi, shape=()):
    return jax.random.uniform(key, shape, minval=lo, maxval=hi)


def _sign(key):
    return jnp.where(jax.random.bernoulli(key), 1.0, -1.0)


def _stbd(h):
    fx, fy = heading_vector(h)
    return fy, -fx


def _at(p, *terms):
    """p + sum of (scale, (vx, vy)) terms."""
    x, y = p
    for s, (vx, vy) in terms:
        x, y = x + s * vx, y + s * vy
    return x, y


def _converge(p, h_a, v_a, h_b, v_b, dist_a, lead_b):
    """Two boats on collision course through point p: A dist_a m before it; B arrives lead_b s earlier (> 0:
    B crosses ahead) or later."""
    tau = dist_a / jnp.maximum(v_a, 0.1)
    fa, fb = heading_vector(h_a), heading_vector(h_b)
    return _at(p, (-v_a * tau, fa)), _at(p, (-v_b * jnp.maximum(tau - lead_b, 1.0), fb))


def _beat_ref(key, k, below_lo, below_hi, lat):
    k1, k2 = jax.random.split(key)
    return _at(k["w"], (-_u(k1, below_lo, below_hi), k["u"]), (_u(k2, -lat, lat), k["r"]))


# --- the situations ------------------------------------------------------------------------------------------
# Each takes (key, ctx) and returns dict(a=(x, y, heading, speed_frac), b=(...), leg, kind, t). ctx holds the
# course k, the line (centre c, direction d pin->committee, half-length hl), L, the close-hauled and
# running TWAs (cs, run) and wind direction wd at the setup's clock (the clock is drawn first: ctx["t"]).


def prestart_entry(key, c):
    """Pre-start, T-80..T-55: boats come in from opposite ends of the line, A on starboard, B on port."""
    ks = jax.random.split(key, 8)
    a = _at(c["c"], (-_u(ks[0], 0.4, 1.0) * c["hl"] - _u(ks[1], 0, 20), c["d"]), (-_u(ks[2], 40, 90), c["n"]))
    b = _at(c["c"], (_u(ks[3], 0.4, 1.0) * c["hl"] + _u(ks[4], 0, 20), c["d"]), (-_u(ks[5], 40, 90), c["n"]))
    ha = wrap_angle(c["wd"] - _u(ks[6], deg(60), deg(110)))
    hb = wrap_angle(c["wd"] + _u(ks[7], deg(60), deg(110)))
    return dict(a=(*a, ha, 0.8), b=(*b, hb, 0.8), leg=0, kind=KIND_START)


def start_leeward(key, c):
    """Final approach, T-30..T-12: both on starboard, B overlapped to windward of A (leeward control)."""
    ks = jax.random.split(key, 6)
    ha = wrap_angle(c["wd"] - _u(ks[0], deg(45), deg(75)))
    a = _at(c["c"], (_u(ks[1], -0.7, 0.7) * c["hl"], c["d"]), (-_u(ks[2], 15, 45), c["n"]))
    b = _at(a, (_u(ks[3], 1.0, 3.0) * c["L"], _stbd(ha)), (_u(ks[4], -1.2, 1.0) * c["L"], heading_vector(ha)))
    hb = wrap_angle(ha + _u(ks[5], -deg(8), deg(8)))
    return dict(a=(*a, ha, 0.6), b=(*b, hb, 0.6), leg=0, kind=KIND_START)


def start_chase(key, c):
    """Pre-start, T-50..T-25: B 2-5 lengths astern of A, chasing it (circling, hunting)."""
    ks = jax.random.split(key, 7)
    ha = wrap_angle(c["wd"] - _sign(ks[0]) * _u(ks[1], deg(60), deg(150)))
    a = _at(c["c"], (_u(ks[2], -0.6, 0.6) * c["hl"], c["d"]), (-_u(ks[3], 30, 70), c["n"]))
    b = _at(a, (-_u(ks[4], 2.0, 5.0) * c["L"], heading_vector(ha)), (_u(ks[5], -1.0, 1.0) * c["L"], _stbd(ha)))
    hb = wrap_angle(ha + _u(ks[6], -deg(15), deg(15)))
    return dict(a=(*a, ha, 0.8), b=(*b, hb, 0.8), leg=0, kind=KIND_START)


def port_starboard(key, c):
    """Beat: A on starboard, B on port, converging; B crosses ahead or behind by up to 1.5 lengths of time."""
    ks = jax.random.split(key, 4)
    p = _beat_ref(ks[0], c["k"], 60, 160, 50)
    ha, hb = wrap_angle(c["wd"] - c["cs"]), wrap_angle(c["wd"] + c["cs"])
    v = c["v_cs"]
    a, b = _converge(p, ha, v, hb, v, _u(ks[1], 3, 12) * c["L"], _u(ks[2], -1.5, 1.5) * c["L"] / v)
    return dict(a=(*a, ha, 1.0), b=(*b, hb, 1.0), leg=None, kind=KIND_LEG)


def same_tack(key, c):
    """Beat, same tack, close: B within 4 lengths ahead/behind and 1.2-5 lengths to windward or leeward of A
    (safe leeward, lee-bow, controlled windward boat)."""
    ks = jax.random.split(key, 6)
    ha = wrap_angle(c["wd"] - _sign(ks[0]) * c["cs"])
    a = _beat_ref(ks[1], c["k"], 70, 170, 60)
    b = _at(a, (_u(ks[2], -4, 4) * c["L"], heading_vector(ha)), (_sign(ks[3]) * _u(ks[4], 1.2, 5) * c["L"], _stbd(ha)))
    return dict(a=(*a, ha, 1.0), b=(*b, wrap_angle(ha + _u(ks[5], -deg(5), deg(5))), 1.0), leg=None, kind=KIND_LEG)


def cover(key, c):
    """Beat: A 3-8 lengths straight upwind of B (B in A's dirty air): covering, tacking duels."""
    ks = jax.random.split(key, 5)
    a = _beat_ref(ks[0], c["k"], 60, 140, 50)
    b = _at(a, (-_u(ks[1], 3, 8) * c["L"], c["k"]["u"]), (_u(ks[2], -1.5, 1.5) * c["L"], c["k"]["r"]))
    ha = wrap_angle(c["wd"] - _sign(ks[3]) * c["cs"])
    hb = wrap_angle(c["wd"] - _sign(ks[4]) * c["cs"])
    return dict(a=(*a, ha, 1.0), b=(*b, hb, 1.0), leg=None, kind=KIND_LEG)


def _wmark_track(key, c, off_lo, off_hi, before_lo, before_hi):
    """A point on a starboard approach that leaves the windward mark to port, off_lo..off_hi lengths to its
    right, before_lo..before_hi lengths before abeam the mark; and the starboard heading."""
    k1, k2 = jax.random.split(key)
    ha = wrap_angle(c["wd"] - c["cs"])
    p = _at(c["k"]["w"], (_u(k1, off_lo, off_hi) * c["L"], _stbd(ha)), (-_u(k2, before_lo, before_hi) * c["L"], heading_vector(ha)))
    return p, ha


def wmark_port_starboard(key, c):
    """Windward mark: A on the starboard layline, B on port coming across close to the mark (RRS 10, 18.1(a),
    18.3 if B tacks in the zone)."""
    ks = jax.random.split(key, 3)
    p, ha = _wmark_track(ks[0], c, 1.0, 2.0, 1.5, 4.0)
    hb = wrap_angle(c["wd"] + c["cs"])
    v = c["v_cs"]
    a, b = _converge(p, ha, v, hb, v, _u(ks[1], 3, 8) * c["L"], _u(ks[2], -1.5, 1.5) * c["L"] / v)
    return dict(a=(*a, ha, 1.0), b=(*b, hb, 1.0), leg=None, kind=KIND_LEG)


def wmark_overlap(key, c):
    """Windward mark: both on the starboard layline, B overlapped inside/outside or just astern (mark-room)."""
    ks = jax.random.split(key, 4)
    p, ha = _wmark_track(ks[0], c, 2.0, 4.0, 4.0, 9.0)
    b = _at(p, (_u(ks[1], -2.5, 1.5) * c["L"], heading_vector(ha)), (_sign(ks[2]) * _u(ks[3], 1.2, 2.5) * c["L"], _stbd(ha)))
    return dict(a=(*p, ha, 1.0), b=(*b, ha, 1.0), leg=None, kind=KIND_LEG)


def _run_ref(key, c, leg, above_lo, above_hi, lat):
    k1, k2 = jax.random.split(key)
    ref = jnp.where(leg == 1, jnp.stack(c["k"]["g"]), jnp.stack(c["k"]["c"]))
    return _at((ref[0], ref[1]), (_u(k1, above_lo, above_hi), c["k"]["u"]), (_u(k2, -lat, lat), c["k"]["r"]))


def run_blanket(key, c, leg):
    """Run: B 2-6 lengths upwind of (astern of) A, able to blanket it; gybes at random."""
    ks = jax.random.split(key, 6)
    a = _run_ref(ks[0], c, leg, 60, 140, 30)
    b = _at(a, (_u(ks[1], 2, 6) * c["L"], c["k"]["u"]), (_u(ks[2], -1, 1) * c["L"], c["k"]["r"]))
    ha = wrap_angle(c["wd"] - _sign(ks[3]) * (c["run"] + _u(ks[4], -deg(10), deg(10))))
    hb = wrap_angle(c["wd"] - _sign(ks[5]) * c["run"])
    return dict(a=(*a, ha, 1.0), b=(*b, hb, 1.0), leg=leg, kind=KIND_LEG)


def run_converge(key, c, leg):
    """Run: A on starboard gybe, B on port gybe, converging (RRS 10 downwind)."""
    ks = jax.random.split(key, 3)
    p = _run_ref(ks[0], c, leg, 50, 120, 30)
    ha, hb = wrap_angle(c["wd"] - c["run"]), wrap_angle(c["wd"] + c["run"])
    v = c["v_run"]
    a, b = _converge(p, ha, v, hb, v, _u(ks[1], 3, 10) * c["L"], _u(ks[2], -1.5, 1.5) * c["L"] / v)
    return dict(a=(*a, ha, 1.0), b=(*b, hb, 1.0), leg=leg, kind=KIND_LEG)


def _run_overlap(key, c, leg, lat):
    ks = jax.random.split(key, 6)
    a = _run_ref(ks[0], c, leg, 4 * c["L"], 10 * c["L"], lat)
    ha = wrap_angle(c["wd"] - _sign(ks[1]) * (c["run"] + _u(ks[2], -deg(10), deg(10))))
    b = _at(a, (_u(ks[3], -2.0, 1.5) * c["L"], heading_vector(ha)), (_sign(ks[4]) * _u(ks[5], 1.2, 3) * c["L"], _stbd(ha)))
    return dict(a=(*a, ha, 1.0), b=(*b, ha, 1.0), leg=leg, kind=KIND_LEG)


def gate_approach(key, c):
    """Leeward gate: 4-10 lengths above it, overlapped or just astern, same gybe (mark-room, gate choice)."""
    return _run_overlap(key, c, 1, 0.8 * c["k"]["hw"])


def finish_approach(key, c):
    """Finish (the start line, downwind): 4-10 lengths above it, overlapped or just astern."""
    return _run_overlap(key, c, 3, 0.6 * c["hl"])


# name -> (sampler, clock range (s, relative to the gun), legs it can be on)
SITUATIONS = {
    "prestart_entry": (prestart_entry, (-80.0, -55.0), (0,)),
    "start_leeward": (start_leeward, (-30.0, -12.0), (0,)),
    "start_chase": (start_chase, (-50.0, -25.0), (0,)),
    "port_starboard": (port_starboard, None, (0, 2)),
    "same_tack": (same_tack, None, (0, 2)),
    "cover": (cover, None, (0, 2)),
    "wmark_port_starboard": (wmark_port_starboard, None, (0, 2)),
    "wmark_overlap": (wmark_overlap, None, (0, 2)),
    "run_blanket": (run_blanket, None, (1, 3)),
    "run_converge": (run_converge, None, (1, 3)),
    "gate_approach": (gate_approach, None, (1,)),
    "finish_approach": (finish_approach, None, (3,)),
}
NAMES = list(SITUATIONS)
FULL_RACE = len(NAMES)  # situation id of the full-race entries
CLOCK = {0: (60.0, 150.0), 1: (200.0, 280.0), 2: (330.0, 420.0), 3: (480.0, 560.0)}  # rough race clock by leg


def target_reached(env, fs, kind, leg0, start_checkpoint, params):
    """(2,) bool: each boat has reached the episode's target and has no penalty left to serve."""
    st = fs.boats
    line = jax.tree.map(lambda x: x[0], st.line)
    cx, cy = 0.5 * (line.pin_x + line.cb_x), 0.5 * (line.pin_y + line.cb_y)
    wd = st.wind.base_dir[0]
    bx, by = bow_position(st.boat, params)
    height = (bx - cx) * jnp.sin(wd) + (by - cy) * jnp.cos(wd)
    start_ok = st.status.started & ~st.status.ocs & (height >= start_checkpoint)
    hit = jnp.select([kind == KIND_LEG, kind == KIND_START], [fs.leg >= leg0 + 1, start_ok], fs.leg >= 4)
    return hit & (fs.penalty_left <= 0.0)


def _place(env, params, fs, spec, t, legs):
    """The race state for a setup: boats placed as spec, started (unless a start), legs, clock t, counters 0."""
    n = 2
    xa, ya, ha, fa = spec["a"]
    xb, yb, hb, fb = spec["b"]
    st = fs.boats
    wd, tws = wind_at(jax.tree.map(lambda x: x[0], st.wind), t)
    h = jnp.stack([ha, hb])
    frac = jnp.stack([jnp.asarray(fa, jnp.float32), jnp.asarray(fb, jnp.float32)])
    v = frac * polar_speed(env.polar, signed_twa(h, wd), tws)
    boat = st.boat.replace(x=jnp.stack([xa, xb]), y=jnp.stack([ya, yb]), heading=h, target_heading=h, speed=v,
                           manoeuvring=jnp.zeros(n, bool))
    start = spec["kind"] == KIND_START
    status = st.status.replace(started=jnp.full(n, not start), ocs=jnp.zeros(n, bool),
                               start_time=jnp.full(n, jnp.inf if start else 0.0))
    st = st.replace(boat=boat, status=status, t=jnp.full(n, t))
    z = jnp.zeros(n, jnp.int32)
    return fs.replace(boats=st, leg=legs, fouls=z, contacts=z, mark_hits=z, shadow_time=jnp.zeros(n))


def sample_situation(env, params, name, key, n):
    """n setups of one situation (before filtering). Returns (fs, kind (n,))."""
    fn, clock, legs = SITUATIONS[name]
    if any_weather(params):
        raise ValueError("the situation library assumes the steady/oscillating wind of the race preset (no weather)")

    def one(k):
        k_reset, k_leg, k_t, k_sit, k_swap = jax.random.split(k, 5)
        _, fs = env.reset(k_reset, params)
        leg = jnp.asarray(legs)[jax.random.randint(k_leg, (), 0, len(legs))]
        if clock is None:
            lo = jnp.select([leg == i for i in range(4)], [CLOCK[i][0] for i in range(4)])
            hi = jnp.select([leg == i for i in range(4)], [CLOCK[i][1] for i in range(4)])
            t = _u(k_t, lo, hi)
        else:
            t = _u(k_t, *clock)
        k = env.course(fs, params)
        line = jax.tree.map(lambda x: x[0], fs.boats.line)
        dx, dy = line.cb_x - line.pin_x, line.cb_y - line.pin_y
        half = 0.5 * jnp.hypot(dx, dy)
        wd, tws = wind_at(jax.tree.map(lambda x: x[0], fs.boats.wind), t)
        cs, run = beat_angle(env.polar, tws), run_angle(env.polar, tws)
        ctx = dict(k=k, c=k["c"], d=(dx / (2 * half), dy / (2 * half)), n=(-dy / (2 * half), dx / (2 * half)), hl=half,
                   L=params.boat_length, wd=wd, cs=cs, run=run,
                   v_cs=polar_speed(env.polar, cs, tws), v_run=polar_speed(env.polar, run, tws))
        spec = fn(k_sit, ctx, leg) if fn in (run_blanket, run_converge) else fn(k_sit, ctx)
        # Boat 0 is A or B at random, so the policy cannot tell the roles of the geometry by its index.
        swap = jax.random.bernoulli(k_swap)
        a, b = spec["a"], spec["b"]
        spec = dict(spec, a=jax.tree.map(lambda p, q: jnp.where(swap, q, p), a, b),
                    b=jax.tree.map(lambda p, q: jnp.where(swap, p, q), a, b))
        return _place(env, params, fs, spec, t, jnp.full(2, leg, jnp.int32))

    fs = jax.jit(jax.vmap(one))(jax.random.split(key, n))
    k0 = KIND_START if SITUATIONS[name][1] is not None else KIND_LEG
    return fs, jnp.full(n, k0, jnp.int32)


def solo_times(env, params, fs, kind, start_checkpoint, steps=SOLO_STEPS):
    """(n, 2) s: time for the digital sailor to reach the target from each setup, sailing each boat alone
    (the other boat is moved far away). inf = not within `steps` s."""
    policy = digital_sailor_agent(env, params)
    key = jax.random.PRNGKey(0)

    def one(fs, kind, b):
        other = 1 - b
        boat = fs.boats.boat
        boat = boat.replace(x=boat.x.at[other].set(PARK_X), y=boat.y.at[other].set(0.0), speed=boat.speed.at[other].set(0.0))
        fs = fs.replace(boats=fs.boats.replace(boat=boat))
        t0, leg0 = fs.boats.t[0], fs.leg

        def body(carry, _):
            fs, reach = carry
            obs = env.get_obs(fs, params)
            a = jax.vmap(policy, in_axes=(None, 0, 0, 0))(key, obs, fs.boats, jnp.arange(2))
            a = jnp.where(jnp.arange(2) == other, HOLD, a)
            _, fs2, _, _, _ = env.step_env(key, fs, a, params)
            hit = target_reached(env, fs2, kind, leg0, start_checkpoint, params)[b]
            reach = jnp.where(jnp.isinf(reach) & hit, fs2.boats.t[0] - t0, reach)
            return (fs2, reach), None

        (_, reach), _ = jax.lax.scan(body, (fs, jnp.float32(jnp.inf)), None, length=steps)
        return reach

    both = jax.vmap(lambda f, k: jnp.stack([one(f, k, 0), one(f, k, 1)]))
    return jax.jit(both)(fs, kind)


def projected_start_times(env, params, fs, start_checkpoint):
    """(n, 2) s: projected undisturbed time to the start target. The digital sailor is no yardstick here: from
    an arbitrary late approach it first sails to its own spot on the line and often starts 20-80 s late. So:
    reach the line at beam-reach speed (not before the gun), then sail start_checkpoint m at best upwind VMG."""
    from sail_env.fleet_rules import point_segment_distance
    from sail_env.polar import beat_vmg

    def one(fs):
        st = fs.boats
        line = jax.tree.map(lambda x: x[0], st.line)
        bx, by = bow_position(st.boat, params)
        d = point_segment_distance(bx, by, line.pin_x, line.pin_y, line.cb_x, line.cb_y)
        tws = st.wind.tws[0]
        t_line = d / polar_speed(env.polar, jnp.radians(90.0), tws)
        return jnp.maximum(t_line, -st.t[0]) + start_checkpoint / beat_vmg(env.polar, tws)

    return jax.jit(jax.vmap(one))(fs)


def build_bank(env, params, key, per_situation, start_checkpoint, full_races=0, names=None, oversample=1.6,
               start_policy="fleet", log=print):
    """The training bank: per_situation valid setups of each situation x the 3 role pairings, then full_races
    race states just after a start (both boats racing). Returns (bank, number of situation entries)."""
    from sail_env.train.curriculum.config_env import make_race_start_bank

    names = NAMES if names is None else names
    parts = []
    for i, name in enumerate(names):
        k = jax.random.fold_in(key, i)
        m = int(np.ceil(per_situation * oversample))
        fs, kind = sample_situation(env, params, name, k, m)
        if SITUATIONS[name][1] is not None:  # a start: projected, not sailed (see projected_start_times)
            t_solo = projected_start_times(env, params, fs, start_checkpoint)
        else:
            t_solo = solo_times(env, params, fs, kind, start_checkpoint)
        apart = ~jax.vmap(lambda b: contacts(b, params.boat_length, 2.0 * params.boat_beam)[0, 1])(fs.boats.boat)
        ok = np.asarray(jnp.all(jnp.isfinite(t_solo), axis=1) & apart)
        idx = np.flatnonzero(ok)[:per_situation]
        if len(idx) < per_situation:
            log(f"  warning: {name}: only {len(idx)} of {m} setups are valid (wanted {per_situation})")
        ts = np.asarray(t_solo)[idx]
        log(f"  {name:22s} {len(idx):5d} setups ({100 * ok.mean():.0f}% valid), undisturbed time "
            f"median {np.median(ts):.0f} s (10-90%: {np.percentile(ts, 10):.0f}-{np.percentile(ts, 90):.0f} s)")
        sid = NAMES.index(name)
        sub = jax.tree.map(lambda x: x[idx], fs)
        for pairing in PAIRINGS:
            parts.append(Bank(fs=sub, role=jnp.tile(jnp.asarray(pairing, jnp.int32), (len(idx), 1)),
                              kind=jnp.asarray(kind)[idx], t_solo=jnp.asarray(ts), situation=jnp.full(len(idx), sid, jnp.int32),
                              leg0=sub.leg))
    # Interleave so that config id = setup x 3 + pairing within each situation (the order does not matter
    # to the sampler; it keeps a setup's three pairings next to each other for reading logs).
    situ = jax.tree.map(lambda *xs: jnp.concatenate(xs), *parts)
    n_sit = int(situ.kind.shape[0])
    if full_races:
        fs = make_race_start_bank(env, params, jax.random.fold_in(key, 999), full_races, start_policy)
        z = jnp.zeros(2, jnp.int32)
        fs = fs.replace(fouls=jnp.zeros_like(fs.fouls), contacts=jnp.zeros_like(fs.contacts),
                        mark_hits=jnp.zeros_like(fs.mark_hits), shadow_time=jnp.zeros_like(fs.shadow_time))
        full = Bank(fs=fs, role=jnp.zeros((full_races, 2), jnp.int32), kind=jnp.full(full_races, KIND_FINISH, jnp.int32),
                    t_solo=jnp.full((full_races, 2), jnp.inf), situation=jnp.full(full_races, FULL_RACE, jnp.int32),
                    leg0=fs.leg + z)
        log(f"  {'full_race':22s} {full_races:5d} race states just after the start")
        situ = jax.tree.map(lambda a, b: jnp.concatenate([a, b]), situ, full)
    return situ, n_sit
