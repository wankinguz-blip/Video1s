"""Math helpers.  Everything that animates is a pure function of loop time t ∈ [0, LOOP)."""
import math

TAU = math.tau
DEG = math.pi / 180.0


def clamp(v, a=0.0, b=1.0):
    return a if v < a else b if v > b else v


def lerp(a, b, u):
    return a + (b - a) * u


def inv_lerp(a, b, v):
    return clamp((v - a) / (b - a))


def smoothstep(a, b, v):
    u = inv_lerp(a, b, v)
    return u * u * (3 - 2 * u)


def smootherstep(a, b, v):
    u = inv_lerp(a, b, v)
    return u * u * u * (u * (u * 6 - 15) + 10)


def ease_out_cubic(u):
    return 1 - (1 - clamp(u)) ** 3


def ease_in_cubic(u):
    return clamp(u) ** 3


def ease_in_out_sine(u):
    return 0.5 - 0.5 * math.cos(math.pi * clamp(u))


def mod(a, n):
    return ((a % n) + n) % n


def wrap_delta(a, b, n):
    """Shortest signed distance from a to b on a circle of circumference n."""
    d = mod(b - a, n)
    return d - n if d > n / 2 else d


def bump(t, c, w, period=None):
    """Smooth bump (0..1) centred at c with half-width w, optionally on a periodic timeline."""
    d = wrap_delta(c, t, period) if period else t - c
    u = abs(d) / w
    if u >= 1:
        return 0.0
    k = 1 - u * u
    return k * k * k


def envelope(t, a, b, rise, fall):
    """Rises over [a, a+rise], holds, falls over [b-fall, b] (smootherstep edges)."""
    return smootherstep(a, a + rise, t) * (1 - smootherstep(b - fall, b, t))


def penvelope(t, a, b, rise, fall, period):
    """Periodic version of envelope: works when [a, b] wraps around the loop boundary."""
    tt = a + mod(t - a, period)
    return envelope(tt, a, b, rise, fall)


# ---------------------------------------------------------------- deterministic randomness
def mulberry32(seed):
    state = [seed & 0xFFFFFFFF]

    def rnd():
        state[0] = (state[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = state[0]
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return rnd


def hash1(n):
    x = ((int(n) ^ 0x9E3779B9) * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    return x / 4294967296.0


def hash2(a, b):
    return hash1(((int(a) * 73856093) ^ (int(b) * 19349663)) & 0x7FFFFFFF)


def loop_noise(t, period, seed, harmonics=(1, 2, 3, 5)):
    """Periodic smooth noise in [-1, 1]: integer harmonics of the loop frequency → exact loop."""
    v = norm = 0.0
    for i, h in enumerate(harmonics):
        amp = 1.0 / h
        v += amp * math.sin(TAU * h * t / period + TAU * hash2(seed, i))
        norm += amp
    return v / norm


# ---------------------------------------------------------------- periodic keyframe splines (C1, loop-closed)
def loop_spline(keys, period):
    """keys: [(time, value), ...] sorted by time in [0, period); value = number or sequence.

    Returns f(t): cardinal (Catmull-Rom, non-uniform) interpolation that wraps around the period,
    so position AND velocity are continuous across the loop boundary.
    """
    n = len(keys)
    times = [float(k[0]) for k in keys]
    vals = [k[1] for k in keys]
    is_seq = hasattr(vals[0], "__len__")
    if is_seq:
        vals = [tuple(float(c) for c in v) for v in vals]

    def at(i):
        j = i % n
        return times[j] + (i // n) * period, vals[j]

    def tangent(i):
        t0, v0 = at(i - 1)
        t2, v2 = at(i + 1)
        dt = t2 - t0
        if is_seq:
            return tuple((b - a) / dt for a, b in zip(v0, v2))
        return (v2 - v0) / dt

    tangents = [tangent(i) for i in range(n)]

    def f(t):
        tt = mod(t, period)
        i = n - 1
        for k in range(n):
            if times[k] > tt:
                i = k - 1
                break
        ta, va = at(i)
        tb, vb = at(i + 1)
        x = tt
        if x < ta:
            x += period
        ma, mb = tangents[i % n], tangents[(i + 1) % n]
        h = tb - ta
        u = (x - ta) / h
        h00 = 2 * u ** 3 - 3 * u ** 2 + 1
        h10 = u ** 3 - 2 * u ** 2 + u
        h01 = -2 * u ** 3 + 3 * u ** 2
        h11 = u ** 3 - u ** 2
        if not is_seq:
            return h00 * va + h10 * h * ma + h01 * vb + h11 * h * mb
        return tuple(h00 * va[k] + h10 * h * ma[k] + h01 * vb[k] + h11 * h * mb[k] for k in range(len(va)))

    return f


def loop_hold_spline(keys, period, ease=0.5):
    """Pose-to-pose helper: like loop_spline but each key is *held* and the move into the next key
    is an ease-in-out occupying the last `ease` fraction of the interval (snappy, readable accents).
    keys: [(time, value)], value = number or sequence."""
    n = len(keys)
    times = [float(k[0]) for k in keys]
    vals = [k[1] for k in keys]
    is_seq = hasattr(vals[0], "__len__")

    def f(t):
        tt = mod(t, period)
        i = n - 1
        for k in range(n):
            if times[k] > tt:
                i = k - 1
                break
        ta, va = times[i % n] + (i // n) * period, vals[i % n]
        tb, vb = times[(i + 1) % n] + ((i + 1) // n) * period, vals[(i + 1) % n]
        x = tt if tt >= ta else tt + period
        u = (x - ta) / (tb - ta)
        w = smootherstep(1 - ease, 1.0, u)
        if not is_seq:
            return va + (vb - va) * w
        return tuple(a + (b - a) * w for a, b in zip(va, vb))

    return f
