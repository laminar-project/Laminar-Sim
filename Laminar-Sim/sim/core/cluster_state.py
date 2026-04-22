# sim/core/cluster_state.py

import math
import bisect
from typing import List


class GlobalMetrics:
    def __init__(self):
        self.total_arrivals = 0
        self.total_execution_starts = 0
        self.total_completions = 0
        self.total_successful_starts = 0
        self.total_pending_expiry = 0
        self.total_duplicate_das = 0
        self.total_uniqueness_violations = 0
        self.total_late_winner_reclaims = 0
        self.total_probe_actions = 0
        self.total_node_score_evals = 0
        self.total_false_optimism_events = 0
        self.total_reservations_granted = 0
        self.total_honest_goodput = 0

        self.reservation_hold_times: List[float] = []
        self.late_winner_residence_times: List[float] = []
        self.control_work_us_array: List[float] = []
        self.start_lats_all: List[float] = []


class ClusterState:
    def __init__(self, num_nodes: int, nodes_per_zone: int = 1024, zone_sizes=None) -> None:
        self.num_nodes = int(num_nodes)
        self.nodes_per_zone = int(max(1, nodes_per_zone))

        if zone_sizes is not None:
            self.zone_sizes: List[int] = [int(x) for x in zone_sizes]
            assert sum(self.zone_sizes) == self.num_nodes, "sum(zone_sizes) must equal num_nodes"
            assert all(x > 0 for x in self.zone_sizes), "all zone sizes must be positive"
            self.num_zones = len(self.zone_sizes)
        else:
            self.num_zones = math.ceil(self.num_nodes / self.nodes_per_zone)
            self.zone_sizes = []
            for zid in range(self.num_zones):
                start = zid * self.nodes_per_zone
                end = min(self.num_nodes, (zid + 1) * self.nodes_per_zone)
                self.zone_sizes.append(end - start)

        self.zone_offsets: List[int] = [0]
        acc = 0
        for sz in self.zone_sizes:
            acc += sz
            self.zone_offsets.append(acc)

        self.FULL_MASK = (1 << 256) - 1
        self.node_masks: List[int] = [self.FULL_MASK for _ in range(self.num_nodes)]

        self.node_s: List[float] = [256.0] * self.num_nodes
        self.node_h: List[float] = [0.0] * self.num_nodes

        self.zone_s: List[float] = [0.0] * self.num_zones
        self.zone_h: List[float] = [0.0] * self.num_zones

        for zid, sz in enumerate(self.zone_sizes):
            self.zone_s[zid] = 256.0 * float(sz)

        self.metrics = GlobalMetrics()
        self.started_logical_tasks = set()

    def zone_of_node(self, node_id: int) -> int:
        return bisect.bisect_right(self.zone_offsets, node_id) - 1

    def allocate(self, node_id: int, demand_mask: int) -> bool:
        if (self.node_masks[node_id] & demand_mask) != demand_mask:
            return False

        self.node_masks[node_id] &= ~demand_mask
        bits = float(demand_mask.bit_count())

        self.node_s[node_id] -= bits
        zid = self.zone_of_node(node_id)
        self.zone_s[zid] -= bits
        return True

    def release(self, node_id: int, demand_mask: int) -> None:
        self.node_masks[node_id] |= demand_mask
        bits = float(demand_mask.bit_count())

        self.node_s[node_id] += bits
        zid = self.zone_of_node(node_id)
        self.zone_s[zid] += bits

    def find_free_mask(self, node_id: int, slots: int, contiguous: bool) -> int:
        free_bits = self.node_masks[node_id]
        if contiguous:
            target = (1 << slots) - 1
            for i in range(257 - slots):
                if (free_bits & (target << i)) == (target << i):
                    return (target << i)
            return 0
        else:
            mask, count = 0, 0
            for i in range(256):
                if (free_bits & (1 << i)):
                    mask |= (1 << i)
                    count += 1
                    if count == slots:
                        break
            return mask if count == slots else 0

    def find_free_mask_from_bits(self, free_bits: int, slots: int, contiguous: bool) -> int:
        free_bits = int(free_bits)
        if contiguous:
            target = (1 << slots) - 1
            for i in range(257 - slots):
                if (free_bits & (target << i)) == (target << i):
                    return (target << i)
            return 0
        else:
            mask, count = 0, 0
            for i in range(256):
                if (free_bits & (1 << i)):
                    mask |= (1 << i)
                    count += 1
                    if count == slots:
                        break
            return mask if count == slots else 0
