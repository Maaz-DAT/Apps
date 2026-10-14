"""
SURV scenario: 20 targets on known tracks + clutter + noise (M-06 step 1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ScenarioTarget:
    track_id: int
    range0_m: float
    az0_rad: float
    speed_m_s: float
    heading_rad: float
    mode3a: int
    fl_ft: float
    address: int
    amplitude: int = 200


def make_scenario_20(seed: int = 42) -> list[ScenarioTarget]:
    """20 targets distributed in range/azimuth with steady tracks."""
    import random
    rng = random.Random(seed)
    targets = []
    for i in range(20):
        targets.append(ScenarioTarget(
            track_id=i + 1,
            range0_m=rng.uniform(8_000, 90_000),
            az0_rad=rng.uniform(0, 2 * math.pi),
            speed_m_s=rng.uniform(80, 250),
            heading_rad=rng.uniform(0, 2 * math.pi),
            mode3a=rng.randint(0o1000, 0o7777),
            fl_ft=rng.choice([0, 5000, 15000, 28000, 33000, 37000]),
            address=0xA00000 + i,
            amplitude=rng.randint(140, 255),
        ))
    return targets


def target_at_time(t: ScenarioTarget, t_s: float) -> tuple[float, float]:
    """Propagate target; return (range_m, az_rad) at time t_s."""
    # simple flat-earth polar motion from initial state
    # convert polar → cartesian, move, convert back
    x = t.range0_m * math.sin(t.az0_rad)
    y = t.range0_m * math.cos(t.az0_rad)
    x += t.speed_m_s * math.sin(t.heading_rad) * t_s
    y += t.speed_m_s * math.cos(t.heading_rad) * t_s
    r = math.hypot(x, y)
    az = math.atan2(x, y) % (2 * math.pi)
    return r, az
