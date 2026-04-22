# sim/mechanisms/zone_layout.py

import math
import random


def generate_hetero_zone_sizes(
    total_nodes: int,
    target_zone_size: int = 1024,
    jitter: float = 0.20,
    seed: int = 0,
):
    """
    Generate a heterogeneous list of zone sizes:
    - All elements are positive integers.
    - sum(zone_sizes) == total_nodes.
    - Mean size approximates target_zone_size.
    - Individual zone size fluctuates between target_zone_size * (1-jitter) and target_zone_size * (1+jitter).
    """
    total_nodes = int(total_nodes)
    target_zone_size = max(1, int(target_zone_size))
    jitter = max(0.0, min(float(jitter), 0.49))

    if total_nodes <= 0:
        return []

    rng = random.Random((seed * 1315423911 + total_nodes * 2654435761 + target_zone_size) & 0xFFFFFFFF)

    num_zones = max(1, int(round(total_nodes / float(target_zone_size))))
    if num_zones > total_nodes:
        num_zones = total_nodes

    base_low = max(1, int(math.floor(target_zone_size * (1.0 - jitter))))
    base_high = max(base_low, int(math.ceil(target_zone_size * (1.0 + jitter))))

    sizes = [rng.randint(base_low, base_high) for _ in range(num_zones)]
    cur = sum(sizes)

    if cur < total_nodes:
        deficit = total_nodes - cur
        for i in range(deficit):
            sizes[i % num_zones] += 1
    elif cur > total_nodes:
        excess = cur - total_nodes
        i = 0
        while excess > 0:
            idx = i % num_zones
            if sizes[idx] > 1:
                sizes[idx] -= 1
                excess -= 1
            i += 1

    rng.shuffle(sizes)

    assert sum(sizes) == total_nodes, f"zone size sum mismatch: {sum(sizes)} != {total_nodes}"
    assert all(x >= 1 for x in sizes), "all zone sizes must be >= 1"

    return sizes
