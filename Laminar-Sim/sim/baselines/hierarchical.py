# sim/baselines/hierarchical.py

import math
import random
from collections import deque, defaultdict
from typing import Optional, Deque, Dict, Tuple, List

from sim.core import EventLoop, ClusterState, JobExecutor
from sim.core.job import JobState
from .config import BaselineConfig


class RayBaseline:
    def __init__(
        self,
        env: EventLoop,
        state: ClusterState,
        executor: JobExecutor,
        cfg: BaselineConfig,
    ) -> None:
        self.env = env
        self.state = state
        self.executor = executor
        self.cfg = cfg

        self.enable_crash = bool(getattr(cfg, "enable_crash", False))
        self.enable_scalar_filter = bool(getattr(cfg, "enable_scalar_filter", True))

        self.metrics_local_service_us: List[float] = []
        self.metrics_gcs_service_us: List[float] = []
        self.metrics_gcs_wait_us: List[float] = []

        self.crashed = False

        # Introduce local queuing on nodes to limit infinite concurrency
        self.local_queues: List[Deque] = [deque() for _ in range(self.state.num_nodes)]
        self.local_busy: List[bool] = [False for _ in range(self.state.num_nodes)]

        self.num_shards = max(1, int(getattr(cfg, "ray_gcs_shards", 4)))
        self.gcs_queues: List[Deque] = [deque() for _ in range(self.num_shards)]
        self.gcs_busy: List[bool] = [False for _ in range(self.num_shards)]
        self.hot_shard = 0

        self.stat_local_attempts = 0
        self.stat_local_hits = 0
        self.stat_local_to_gcs = 0
        self.stat_local_commit_failures = 0
        self.stat_gcs_enqueues = 0
        self.stat_gcs_process = 0
        self.stat_gcs_hits = 0
        self.stat_gcs_commit_failures = 0
        self.stat_spillbacks = 0
        self.stat_drops = 0

        # Core mechanism: Stale view and heartbeat synchronization
        self.gcs_view_s = list(self.state.node_s)
        self.gcs_view_masks = list(self.state.node_masks)
        
        self.heartbeat_ms = float(getattr(self.cfg, "baseline_heartbeat_ms", 50.0))
        self.env.schedule(self.heartbeat_ms, self._sync_gcs_state)

        # Core mechanism: Raylet local mutex queues
        self.raylet_commit_queues: List[Deque] = [deque() for _ in range(self.state.num_nodes)]
        self.raylet_commit_busy: List[bool] = [False for _ in range(self.state.num_nodes)]
        
        # Core mechanism: Local pending buffers to prevent self-collision
        self.node_pending_s: List[float] = [0.0 for _ in range(self.state.num_nodes)]
        self.node_pending_masks: List[int] = [0 for _ in range(self.state.num_nodes)]

    def _sync_gcs_state(self) -> None:
        if self.crashed or getattr(self.env, "crashed", False):
            return
        self.gcs_view_s = list(self.state.node_s)
        self.gcs_view_masks = list(self.state.node_masks)
        self.env.schedule(self.heartbeat_ms, self._sync_gcs_state)

    def submit_job(self, job) -> None:
        if self.crashed:
            return

        if getattr(job, "retries", None) is None:
            job.retries = 0
        if not hasattr(job, "ray_tried_zones"):
            job.ray_tried_zones = set()

        entry_node = getattr(job, "entry_node", None)
        if entry_node is None:
            entry_node = random.randrange(self.state.num_nodes)
            job.entry_node = entry_node

        if len(self.local_queues[entry_node]) >= 1000:
            # Fast-drop to propagate congestion pressure to the central GCS
            self.stat_drops += 1
            job.state = JobState.KILLED
            if hasattr(self, "executor"): self.executor.num_killed += 1
            return
        
        self.local_queues[entry_node].append(job)

        if not self.local_busy[entry_node]:
            self.local_busy[entry_node] = True
            self.env.schedule(0.0, self._process_local, entry_node)

    def submit(self, job) -> None:
        self.submit_job(job)
    
    def _maybe_crash(self, decision_us: float) -> bool:
        if not self.enable_crash:
            return False
        timeout_us = float(getattr(self.cfg, "timeout_crash_us", 500000.0))
        if decision_us <= timeout_us:
            return False

        self.crashed = True
        for q in self.gcs_queues:
            while q:
                j = q.popleft()
                if getattr(j, "state", None) not in (JobState.COMPLETED, JobState.KILLED):
                    j.state = JobState.KILLED
                    self.executor.num_killed += 1
        return True
    
    def _job_needs_contig(self, job) -> bool:
        return getattr(job, "job_type", None) in ("LLM_WHALE", "WHALE", "RED_SQUATTER")

    def _mask_for_node(self, nid: int, job) -> int:
        if self.enable_scalar_filter and getattr(self.state, "node_s", None) is not None:
            if float(self.state.node_s[nid]) < float(job.mass):
                return 0

        return int(
            self.state.find_free_mask_from_bits(
                int(self.state.node_masks[nid]),
                int(job.mass),
                self._job_needs_contig(job),
            )
        )

    def _sample_candidate_nodes(self, start: int, end: int, k: int) -> List[int]:
        if end <= start:
            return []
        nodes = list(range(start, end))
        if len(nodes) <= k:
            random.shuffle(nodes)
            return nodes
        return random.sample(nodes, max(1, k))

    def _pick_feasible_from_sample(
        self, start: int, end: int, job, k: int, is_gcs: bool = False
    ) -> Tuple[Optional[int], int]:
        candidates = self._sample_candidate_nodes(start, end, max(1, k))
        best_node = None
        best_mask = 0
        best_score = -float("inf")

        for nid in candidates:
            if is_gcs:
                s_val = float(self.gcs_view_s[nid])
                mask_val = int(self.gcs_view_masks[nid])
            else:
                s_val = max(0.0, float(self.state.node_s[nid]) - float(self.node_pending_s[nid]))
                mask_val = int(self.state.node_masks[nid]) | int(self.node_pending_masks[nid])

            if self.enable_scalar_filter and s_val < float(job.mass):
                continue

            free_mask = int(
                self.state.find_free_mask_from_bits(
                    mask_val, int(job.mass), self._job_needs_contig(job)
                )
            )

            if free_mask == 0:
                continue

            score = (1.0 + s_val) / (1.0 + float(self.state.node_h[nid]))
            if score > best_score:
                best_score = score
                best_node = nid
                best_mask = free_mask

        if best_node is not None:
            if is_gcs:
                self.gcs_view_s[best_node] = max(0.0, float(self.gcs_view_s[best_node]) - float(job.mass))
                self.gcs_view_masks[best_node] = int(self.gcs_view_masks[best_node]) & (~best_mask)
            else:
                self.node_pending_s[best_node] += float(job.mass)
                self.node_pending_masks[best_node] |= best_mask

        return best_node, best_mask

    def _process_local(self, nid: int) -> None:
        if self.crashed:
            self.local_busy[nid] = False
            return
        
        q = self.local_queues[nid]
        if not q:
            self.local_busy[nid] = False
            return

        job = q.popleft()
        
        timeout_us = float(getattr(self.cfg, "timeout_crash_us", 500000.0))
        if (self.env.now - job.arrival_time) * 1000.0 > timeout_us:
            self.stat_drops += 1
            job.state = JobState.KILLED
            if hasattr(self, "executor"): self.executor.num_killed += 1
            if q: self.env.schedule(0.0, self._process_local, nid)
            else: self.local_busy[nid] = False
            return

        self.stat_local_attempts += 1

        zid = self.state.zone_of_node(nid)
        job.ray_tried_zones.add(zid)
        start, end = nid, nid + 1 
        k = 1

        alpha_loc = float(getattr(self.cfg, "ray_local_base_us", 1.0))
        beta_node = float(getattr(self.cfg, "slurm_per_node_us", 0.0001))

        sampled = max(1, min(k, max(1, end - start)))
        congestion = 1.0 + len(q) / 100.0
        
        # Inject system jitter to avoid artificial constant queuing assumption
        base_decision = (alpha_loc + beta_node * sampled) * congestion
        decision_us = base_decision * random.uniform(0.8, 1.2)

        # Record Local service time independently
        self.metrics_local_service_us.append(float(decision_us))
        
        if self._maybe_crash(decision_us):
            self.local_busy[nid] = False
            return

        node, mask = self._pick_feasible_from_sample(start, end, job, k, is_gcs=False)
        
        self.env.schedule(decision_us / 1000.0, self._finish_local, nid, job, node, mask)

    def _finish_local(self, nid: int, job, target_node: Optional[int], target_mask: int) -> None:
        if self.crashed:
            if target_node is not None:
                self.node_pending_s[target_node] = max(0.0, self.node_pending_s[target_node] - float(job.mass))
                self.node_pending_masks[target_node] &= ~target_mask
            self.local_busy[nid] = False
            return
            
        net_rtt_us = float(getattr(self.cfg, "ray_network_rtt_us", 50.0))
        if target_node is not None:
            local_ipc_ms = 0.1 * random.uniform(0.5, 2.0)
            self.env.schedule(local_ipc_ms, self._commit_local, job, target_node, target_mask)
        else:
            self.stat_local_to_gcs += 1
            self.env.schedule(net_rtt_us / 1000.0, self._enqueue_gcs, job)

        if self.local_queues[nid]:
            self.env.schedule(0.0, self._process_local, nid)
        else:
            self.local_busy[nid] = False

    def _commit_local(self, job, node: int, mask: int) -> None:
        timeout_us = float(getattr(self.cfg, "timeout_crash_us", 500000.0))
        is_timeout = (self.env.now - job.arrival_time) * 1000.0 > timeout_us
        
        if is_timeout or self.crashed or getattr(job, "state", None) == JobState.KILLED:
            # Exception cleanup: must release pending compute resources
            self.node_pending_s[node] = max(0.0, self.node_pending_s[node] - float(job.mass))
            self.node_pending_masks[node] &= ~mask
            
            if is_timeout and getattr(job, "state", None) != JobState.KILLED:
                self.stat_drops += 1
                job.state = JobState.KILLED
                if hasattr(self, "executor"): self.executor.num_killed += 1
            return

        self.raylet_commit_queues[node].append(("LOCAL", job, mask))
        if not self.raylet_commit_busy[node]:
            self.raylet_commit_busy[node] = True
            self.env.schedule(0.0, self._process_raylet_lock, node)

    def _choose_shard(self, job=None) -> int:
        if self.num_shards == 1: return 0
        entry_node = getattr(job, "entry_node", None)
        if entry_node is not None: return entry_node % self.num_shards

        hot_prob = float(getattr(self.cfg, "ray_gcs_shard_bias", 0.5))
        hot_prob = min(max(hot_prob, 0.0), 1.0)
        if random.random() < hot_prob: return self.hot_shard
        return random.randrange(self.num_shards)

    def _enqueue_gcs(self, job) -> None:
        if self.crashed: return
            
        timeout_us = float(getattr(self.cfg, "timeout_crash_us", 500000.0))
        if (self.env.now - job.arrival_time) * 1000.0 > timeout_us:
            self.stat_drops += 1
            job.state = JobState.KILLED
            if hasattr(self, "executor"): self.executor.num_killed += 1
            return

        shard = self._choose_shard(job)
        max_q = int(getattr(self.cfg, "ray_gcs_max_queue", 200000))
        
        if len(self.gcs_queues[shard]) >= max_q:
            self.stat_drops += 1
            job.state = JobState.KILLED
            if hasattr(self, "executor"): self.executor.num_killed += 1
            return
            
        self.stat_gcs_enqueues += 1
        # Instrumentation: record the physical time of entering the GCS queue
        job._gcs_enqueue_time = self.env.now  
        self.gcs_queues[shard].append(job)

        if not self.gcs_busy[shard]:
            self.gcs_busy[shard] = True
            self.env.schedule(0.0, self._process_gcs, shard)

    def _zone_score(self, zid: int, job) -> float:
        start = int(self.state.zone_offsets[zid])
        end = (
            int(self.state.zone_offsets[zid + 1])
            if zid + 1 < len(self.state.zone_offsets)
            else self.state.num_nodes
        )

        zs = sum(self.gcs_view_s[start:end])
        zh = float(self.state.zone_h[zid])

        if self._job_needs_contig(job):
            max_node_s = 0.0
            for nid in range(start, end):
                if float(self.gcs_view_s[nid]) > max_node_s:
                    max_node_s = float(self.gcs_view_s[nid])
            if max_node_s < float(job.mass):
                return -float("inf")
        else:
            if zs < float(job.mass):
                return -float("inf")

        return (1.0 + zs) / (1.0 + zh)

    def _best_zone_for_job(self, job) -> Optional[int]:
        num_zones = max(1, int(self.state.num_zones))
        tried = getattr(job, "ray_tried_zones", set())

        best_z = None
        best_score = -float("inf")

        for zid in range(num_zones):
            if zid in tried:
                continue
            score = self._zone_score(zid, job)
            if score > best_score:
                best_score = score
                best_z = zid

        if best_z is not None:
            return best_z

        for zid in range(num_zones):
            score = self._zone_score(zid, job)
            if score > best_score:
                best_score = score
                best_z = zid

        return best_z

    def _process_gcs(self, shard: int) -> None:
        if self.crashed:
            self.gcs_busy[shard] = False
            return

        q = self.gcs_queues[shard]
        if not q:
            self.gcs_busy[shard] = False
            return

        q_len = len(q)

        alpha_gcs = float(getattr(self.cfg, "ray_gcs_base_us", 50.0))
        beta_node = float(getattr(self.cfg, "slurm_per_node_us", 0.0001))

        scale = float(getattr(self.cfg, "ray_gcs_hotspot_scale", 4000.0))
        N = q_len / max(1.0, scale)

        alpha_contention = 0.5
        beta_coherency = 1.2
        usl_penalty = 1.0 + alpha_contention * N + beta_coherency * (N ** 2)
        usl_penalty = min(usl_penalty, 30.0)

        decision_us = (alpha_gcs + self.state.num_nodes * beta_node) * usl_penalty

        self.stat_gcs_process += 1
        self.metrics_gcs_service_us.append(float(decision_us))

        if self._maybe_crash(decision_us):
            self.gcs_busy[shard] = False
            return

        self.env.schedule(decision_us / 1000.0, self._finish_gcs, shard)

    def _delayed_requeue_gcs(self, job, shard: int) -> None:
        if self.crashed:
            return
        # Rewrite the timestamp upon re-queue to prevent immediate timeout
        job._gcs_enqueue_time = self.env.now  
        self.gcs_queues[shard].append(job)
        if not self.gcs_busy[shard]:
            self.gcs_busy[shard] = True
            self.env.schedule(0.0, self._process_gcs, shard)

    def _finish_gcs(self, shard: int) -> None:
        if self.crashed:
            self.gcs_busy[shard] = False
            return

        q = self.gcs_queues[shard]
        if not q:
            self.gcs_busy[shard] = False
            return

        job = q.popleft()
        
        # Calculate GCS queuing wait time (W_q) in microseconds upon dequeue
        wait_us = (self.env.now - getattr(job, "_gcs_enqueue_time", self.env.now)) * 1000.0
        self.metrics_gcs_wait_us.append(wait_us)
        
        timeout_us = float(getattr(self.cfg, "timeout_crash_us", 500000.0))
        if (self.env.now - job.arrival_time) * 1000.0 > timeout_us:
            self.stat_drops += 1
            job.state = JobState.KILLED
            if hasattr(self, "executor"): self.executor.num_killed += 1
            if self.gcs_queues[shard]:
                self.env.schedule(0.0, self._process_gcs, shard)
            else:
                self.gcs_busy[shard] = False
            return

        zid = self._best_zone_for_job(job)
        if zid is not None:
            job.ray_tried_zones.add(zid)
            start = int(self.state.zone_offsets[zid])
            end = (
                int(self.state.zone_offsets[zid + 1])
                if zid + 1 < len(self.state.zone_offsets)
                else self.state.num_nodes
            )
            k = int(getattr(self.cfg, "ray_local_k", 32)) * 2
            node, mask = self._pick_feasible_from_sample(start, end, job, k, is_gcs=True)
            net_rtt_us = float(getattr(self.cfg, "ray_network_rtt_us", 50.0))
            if node is not None:
                self.env.schedule(net_rtt_us / 1000.0, self._commit_gcs, shard, job, node, mask)
            else:
                self.env.schedule(net_rtt_us / 1000.0, self._spillback_or_drop, job)
        else:
            self._spillback_or_drop(job)

        if self.gcs_queues[shard]:
            self.env.schedule(0.0, self._process_gcs, shard)
        else:
            self.gcs_busy[shard] = False

    def _commit_gcs(self, shard: int, job, node: int, mask: int) -> None:
        timeout_us = float(getattr(self.cfg, "timeout_crash_us", 500000.0))
        if (self.env.now - job.arrival_time) * 1000.0 > timeout_us:
            self.stat_drops += 1
            job.state = JobState.KILLED
            if hasattr(self, "executor"): self.executor.num_killed += 1
            return 

        if self.crashed or getattr(job, "state", None) == JobState.KILLED:
            return 

        self.raylet_commit_queues[node].append(("GCS", job, mask))
        if not self.raylet_commit_busy[node]:
            self.raylet_commit_busy[node] = True
            self.env.schedule(0.0, self._process_raylet_lock, node)

    def _process_raylet_lock(self, nid: int) -> None:
        if self.crashed:
            self.raylet_commit_busy[nid] = False
            return
            
        q = self.raylet_commit_queues[nid]
        if not q:
            self.raylet_commit_busy[nid] = False
            return

        source, job, mask = q.popleft()
        
        lock_cost_us = float(getattr(self.cfg, "raylet_lock_cost_us", 50.0))
        actual_lock_us = lock_cost_us * random.uniform(0.8, 1.2)

        self.env.schedule(actual_lock_us / 1000.0, self._finish_raylet_lock, nid, source, job, mask)

    def _finish_raylet_lock(self, nid: int, source: str, job, mask: int) -> None:
        timeout_us = float(getattr(self.cfg, "timeout_crash_us", 500000.0))
        if (self.env.now - job.arrival_time) * 1000.0 > timeout_us:
            if getattr(job, "state", None) != JobState.KILLED:
                self.stat_drops += 1
                job.state = JobState.KILLED
                if hasattr(self, "executor"): self.executor.num_killed += 1

        # Absolute defense: ensure pending capacity is released regardless of success or failure
        if source == "LOCAL":
            self.node_pending_s[nid] = max(0.0, self.node_pending_s[nid] - float(job.mass))
            self.node_pending_masks[nid] &= ~mask

        if self.crashed or getattr(job, "state", None) == JobState.KILLED:
            pass 
        else:
            if mask != 0 and (self.state.node_masks[nid] & mask) == mask:
                if source == "LOCAL":
                    self.stat_local_hits += 1
                else:
                    self.stat_gcs_hits += 1
                    
                job.demand_mask = mask  
                self.state.allocate(nid, job.demand_mask)
                self.executor.start_job(job, nid, already_allocated=True)
            else:
                if source == "LOCAL":
                    self.stat_local_commit_failures += 1
                else:
                    self.stat_gcs_commit_failures += 1
                self._enqueue_gcs(job)

        if self.raylet_commit_queues[nid]:
            self.env.schedule(0.0, self._process_raylet_lock, nid)
        else:
            self.raylet_commit_busy[nid] = False

    def _spillback_or_drop(self, job) -> None:
        job.retries = getattr(job, "retries", 0)
        max_spillbacks = int(getattr(self.cfg, "ray_max_spillbacks", 4))
        base_ms = float(getattr(self.cfg, "ray_retry_base_ms", 2.0))

        if job.retries < max_spillbacks:
            self.stat_spillbacks += 1
            job.retries += 1
            backoff_ms = base_ms * (2 ** min(job.retries, 8))
            en = getattr(job, "entry_node", 0)
            self.env.schedule(backoff_ms, self._resubmit_local, en, job)
            return

        self.stat_drops += 1
        job.state = JobState.KILLED
        if hasattr(self, "executor"): self.executor.num_killed += 1

    def _resubmit_local(self, en, job):
        if self.crashed: return
        
        timeout_us = float(getattr(self.cfg, "timeout_crash_us", 500000.0))
        if (self.env.now - job.arrival_time) * 1000.0 > timeout_us:
            self.stat_drops += 1
            job.state = JobState.KILLED
            if hasattr(self, "executor"): self.executor.num_killed += 1
            return

        if len(self.local_queues[en]) >= 1000:
            self.stat_drops += 1
            job.state = JobState.KILLED
            if hasattr(self, "executor"): self.executor.num_killed += 1
            return
            
        self.local_queues[en].append(job)
        if not self.local_busy[en]:
            self.local_busy[en] = True
            self.env.schedule(0.0, self._process_local, en)


class FluxBaseline:
    """
    Flux-like hierarchical broker tree scheduler.
    """

    def __init__(
        self,
        env: EventLoop,
        state: ClusterState,
        executor: JobExecutor,
        cfg: BaselineConfig,
    ) -> None:
        self.env = env
        self.state = state
        self.executor = executor
        self.cfg = cfg

        self.enable_crash = bool(getattr(cfg, "enable_crash", False))
        self.enable_scalar_filter = bool(getattr(cfg, "enable_scalar_filter", True))

        self.metrics_decision_us: List[float] = []
        self.crashed = False

        self.tree_fanout = max(2, int(getattr(cfg, "flux_tree_fanout", 16)))
        
        # Physical reality: A leaf broker maps to a physical rack managing leaf_k nodes.
        target_leaf_size = max(1, int(getattr(self.cfg, "flux_leaf_k", 32)))
        
        # Calculate the required number of leaf nodes based on cluster size
        num_target_leaves = math.ceil(self.state.num_nodes / target_leaf_size)
        
        # Infer the true physical depth of the tree
        self.tree_depth = max(2, 1 + math.ceil(math.log(max(2, num_target_leaves), self.tree_fanout)))
        self.leaf_level = self.tree_depth - 1

        self.queues: Dict[Tuple[int, int, int], Deque] = defaultdict(deque)
        self.busy: Dict[Tuple[int, int, int], bool] = defaultdict(bool)

        self.parent_map: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        self.leaf_keys: List[Tuple[int, int, int]] = []
        self._build_tree(0, 0, self.state.num_nodes)

        # Flux global stale view and heartbeat synchronization
        self.flux_view_s = list(self.state.node_s)
        self.flux_view_h = list(self.state.node_h)
        self.flux_view_masks = list(self.state.node_masks)
        
        self.heartbeat_ms = float(getattr(self.cfg, "baseline_heartbeat_ms", 50.0))
        self.env.schedule(self.heartbeat_ms, self._sync_flux_state)
        
        # Independent, isolated pending buffers per Broker key
        # Leaf nodes are physically isolated from each other
        self.broker_pending_s: Dict[Tuple[int,int,int], List[float]] = defaultdict(lambda: [0.0]*self.state.num_nodes)
        self.broker_pending_masks: Dict[Tuple[int,int,int], List[int]] = defaultdict(lambda: [0]*self.state.num_nodes)

    # Heartbeat synchronization thread
    def _sync_flux_state(self) -> None:
        if self.crashed or getattr(self.env, "crashed", False):
            return
        self.flux_view_s = list(self.state.node_s)
        self.flux_view_h = list(self.state.node_h)
        self.flux_view_masks = list(self.state.node_masks)
        self.env.schedule(self.heartbeat_ms, self._sync_flux_state)

    # Entry point: distributed leaf-node injection
    def submit_job(self, job) -> None:
        if self.crashed: return
        if getattr(job, "retries", None) is None: job.retries = 0
        job.flux_tried_regions = set()
        
        leaf_key = self._choose_initial_leaf_key(job) 
        self._enqueue_broker(*leaf_key, job)
    
    def submit(self, job) -> None:
        self.submit_job(job)
        

    def _maybe_crash(self, decision_us: float) -> bool:
        if not self.enable_crash:
            return False

        # Unify default timeout to 500ms
        timeout_us = float(getattr(self.cfg, "timeout_crash_us", 500000.0))
        if decision_us <= timeout_us:
            return False

        self.crashed = True
        for q in self.queues.values():
            while q:
                j = q.popleft()
                if getattr(j, "state", None) not in (JobState.COMPLETED, JobState.KILLED):
                    j.state = JobState.KILLED
                    if hasattr(self, "executor"): self.executor.num_killed += 1
        return True

    def _job_needs_contig(self, job) -> bool:
        return getattr(job, "job_type", None) in ("LLM_WHALE", "WHALE", "RED_SQUATTER")

    def _split_children(self, start: int, end: int) -> List[Tuple[int, int]]:
        span = max(0, end - start)
        if span <= 1:
            return [(start, end)] if end > start else []

        child_count = min(self.tree_fanout, span)
        step = math.ceil(span / child_count)

        children = []
        cur = start
        for _ in range(child_count):
            if cur >= end:
                break
            nxt = min(end, cur + step)
            if cur < nxt:
                children.append((cur, nxt))
            cur = nxt
        return children

    def _build_tree(self, level: int, start: int, end: int) -> None:
        if level == self.leaf_level:
            self.leaf_keys.append((level, start, end))
            return

        for cstart, cend in self._split_children(start, end):
            self.parent_map[(level + 1, cstart, cend)] = (level, start, end)
            self._build_tree(level + 1, cstart, cend)

    def _choose_initial_leaf_key(self, job=None) -> Tuple[int, int, int]:
        if not self.leaf_keys:
            return (self.leaf_level, 0, self.state.num_nodes)

        anchor = getattr(job, "entry_node", None)
        if anchor is None:
            return random.choice(self.leaf_keys)

        start, end = 0, self.state.num_nodes
        for level in range(1, self.leaf_level + 1):
            chosen = None
            for cstart, cend in self._split_children(start, end):
                if cstart <= anchor < cend:
                    chosen = (cstart, cend)
                    break
            if chosen is None:
                break
            start, end = chosen

        return (self.leaf_level, start, end)
    
    def _find_parent_key(self, level: int, start: int, end: int) -> Optional[Tuple[int, int, int]]:
        return self.parent_map.get((level, start, end), None)

    def _mask_for_node(self, nid: int, job) -> int:
        if self.enable_scalar_filter and getattr(self.state, "node_s", None) is not None:
            if float(self.state.node_s[nid]) < float(job.mass):
                return 0

        return int(
            self.state.find_free_mask_from_bits(
                int(self.state.node_masks[nid]),
                int(job.mass),
                self._job_needs_contig(job),
            )
        )

    # Queues: preserve hierarchical capacity limits and TTL for backpressure
    def _enqueue_broker(self, level: int, start: int, end: int, job) -> None:
        if self.crashed: return
        
        # Interception: Entry timeout check
        timeout_us = float(getattr(self.cfg, "timeout_crash_us", 500000.0))
        if (self.env.now - job.arrival_time) * 1000.0 > timeout_us:
            job.state = JobState.KILLED
            if hasattr(self, "executor"): self.executor.num_killed += 1
            return
            
        key = (level, start, end)
        
        root_max = int(getattr(self.cfg, "flux_root_max_queue", 200000))
        level_max = max(1000, root_max // (self.tree_fanout ** level))
        
        # Enforce strict broker queue limits
        if len(self.queues[key]) >= level_max:
            # Physical reality: Target broker buffer full, causing network layer DoS.
            # The sender is blocked and times out,
            # then forced to fallback via _handle_broker_fail and requeue from Root.
            
            # Simulate communication layer congestion timeout (e.g., 50ms block)
            zmq_timeout_ms = 50.0 
            
            self.env.schedule(zmq_timeout_ms, self._handle_broker_fail, job, level, start, end)
            return

        self.queues[key].append(job)
        if not self.busy[key]:
            self.busy[key] = True
            self.env.schedule(0.0, self._process_broker, level, start, end)

    def _process_broker(self, level: int, start: int, end: int) -> None:
        key = (level, start, end)

        if self.crashed:
            self.busy[key] = False
            return

        q = self.queues[key]
        if not q:
            self.busy[key] = False
            return

        q_len = len(q)

        # Get base parameters
        base_us = float(getattr(self.cfg, "flux_match_base_us", 2.5))
        per_node_us = float(getattr(self.cfg, "flux_per_node_us", 0.001))

        # Tree scheduling is essentially O(fanout)
        if level < self.leaf_level:
            scanned_units = min(self.tree_fanout, max(1, end - start))
        else:
            scanned_units = min(max(1, end - start), int(getattr(self.cfg, "flux_leaf_k", 32)))

        # Reinstall Universal Scalability Law (USL) queuing degradation
        if level == 0:
            # Only the Root node suffers the global concurrency penalty
            scale = float(getattr(self.cfg, "flux_root_scale", 4000.0))
            # Performance degrades exponentially as queue exceeds scale capacity
            congestion = math.exp(min(q_len / max(1.0, scale), 10.0))
        else:
            # Normal nodes only incur minor queuing overhead
            congestion = 1.0 + min(q_len / 256.0, 5.0) * 0.1

        # Compute real cost = (base + scan) * congestion penalty
        decision_us = (base_us + per_node_us * scanned_units) * congestion
        self.metrics_decision_us.append(float(decision_us))

        if self._maybe_crash(decision_us):
            self.busy[key] = False
            return

        self.env.schedule(decision_us / 1000.0, self._finish_broker, level, start, end)

    def _subtree_summary(self, start: int, end: int, job) -> Tuple[bool, float]:
        total_s = 0.0
        total_h = 0.0
        max_node_s = 0.0

        for nid in range(start, end):
            # Physical constraint: Broker relies on stale tree-network broadcasts
            s = float(self.flux_view_s[nid])
            h = float(self.flux_view_h[nid])
            total_s += s
            total_h += h
            if s > max_node_s:
                max_node_s = s

        feasible = max_node_s >= float(job.mass)

        if not feasible:
            return False, -float("inf")

        score = (1.0 + total_s) / (1.0 + total_h)
        return True, score

    def _rank_children(
        self,
        level: int,
        start: int,
        end: int,
        job,
    ) -> List[Tuple[float, Tuple[int, int, int]]]:
        ranked = []

        for cstart, cend in self._split_children(start, end):
            child_key = (level + 1, cstart, cend)
            if child_key in getattr(job, "flux_tried_regions", set()):
                continue

            feasible, score = self._subtree_summary(cstart, cend, job)
            if feasible:
                # Inject jitter: randomly fluctuate score by +/- 20% to spread tasks and prevent thundering herds
                jitter_score = score * random.uniform(0.8, 1.2)
                ranked.append((jitter_score, child_key))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked

    def _probe_leaf_nodes(self, start: int, end: int, job) -> Tuple[Optional[int], int]:
        span = max(0, end - start)
        if span <= 0:
            return None, 0

        k = int(getattr(self.cfg, "flux_leaf_k", 32))
        candidates = list(range(start, end))
        
        # Physical jitter: shuffle traversal order to prevent single-node bottlenecks
        random.shuffle(candidates)
        if len(candidates) > k:
            candidates = candidates[:k]

        broker_key = (start, end) # Use managed interval as unique ID for this Broker

        for nid in candidates:
            # Physical constraint: view is stale, and only tracks this Broker's own pending allocations
            pending_s = float(self.broker_pending_s[broker_key][nid])
            pending_mask = int(self.broker_pending_masks[broker_key][nid])

            s_val = max(0.0, float(self.flux_view_s[nid]) - pending_s)
            
            # Bitwise merge: stale map bits | this Broker's newly allocated bits
            mask_val = int(self.flux_view_masks[nid]) | pending_mask
            
            if self.enable_scalar_filter and s_val < float(job.mass):
                continue

            free_mask = int(
                self.state.find_free_mask_from_bits(
                    mask_val, int(job.mass), self._job_needs_contig(job)
                )
            )

            if free_mask != 0:
                # Core constraint: do not modify global flux_view; only update Broker's private ledger
                self.broker_pending_s[broker_key][nid] += float(job.mass)
                self.broker_pending_masks[broker_key][nid] |= free_mask
                return nid, free_mask

        return None, 0

    def _delayed_requeue_broker(self, job, level: int, start: int, end: int) -> None:
        if self.crashed:
            return
        self._enqueue_broker(level, start, end, job)

    def _finish_broker(self, level: int, start: int, end: int) -> None:
        key = (level, start, end)

        if self.crashed:
            self.busy[key] = False
            return

        q = self.queues[key]
        if not q:
            self.busy[key] = False
            return

        job = q.popleft()
        
        # Flux Broker queue exit timeout check
        timeout_us = float(getattr(self.cfg, "timeout_crash_us", 500000.0))
        if (self.env.now - job.arrival_time) * 1000.0 > timeout_us:
            job.state = JobState.KILLED
            if hasattr(self, "executor"): self.executor.num_killed += 1
            if self.queues[key]:
                self.env.schedule(0.0, self._process_broker, level, start, end)
            else:
                self.busy[key] = False
            return
            
        hop_ms = float(getattr(self.cfg, "flux_hop_delay_us", 10.0)) / 1000.0

        if level < self.leaf_level:
            ranked = self._rank_children(level, start, end, job)
            if ranked:
                _, child_key = ranked[0]
                job.flux_tried_regions.add(child_key)
                self.env.schedule(hop_ms, self._enqueue_broker, *child_key, job)
            else:
                parent_key = self._find_parent_key(level, start, end)
                if parent_key is None:
                    self._handle_broker_fail(job, level, start, end)
                else:
                    self.env.schedule(hop_ms, self._enqueue_broker, *parent_key, job)
        else:
            node, mask = self._probe_leaf_nodes(start, end, job)
            if node is not None:
                self.env.schedule(hop_ms, self._commit_start, job, node, mask, level, start, end)
            else:
                parent_key = self._find_parent_key(level, start, end)
                if parent_key is None:
                    self._handle_broker_fail(job, level, start, end)
                else:
                    self.env.schedule(hop_ms, self._enqueue_broker, *parent_key, job)

        if self.queues[key]:
            self.env.schedule(0.0, self._process_broker, level, start, end)
        else:
            self.busy[key] = False

    def _handle_broker_fail(self, job, level: int, start: int, end: int) -> None:
        max_retries = int(getattr(self.cfg, "flux_max_retries", 3))
        base_ms = float(getattr(self.cfg, "flux_retry_base_ms", 2.0))

        if getattr(job, "retries", 0) < max_retries:
            job.retries += 1
            job.flux_tried_regions = set()
            backoff_ms = base_ms * (2 ** min(job.retries, 8))
            
            # Failed and retried tasks must re-queue from the Root for centralized dispatch
            root_key = (0, 0, self.state.num_nodes)
            self.env.schedule(backoff_ms, self._delayed_requeue_broker, job, *root_key)
            return

        job.state = JobState.KILLED
        if hasattr(self, "executor"): self.executor.num_killed += 1

    def _commit_start(self, job, node: int, mask: int, level: int, start: int, end: int) -> None:
        # Garbage Collection: immediately return resources to leaf Broker's private ledger
        broker_key = (start, end)
        self.broker_pending_s[broker_key][node] = max(0.0, self.broker_pending_s[broker_key][node] - float(job.mass))
        self.broker_pending_masks[broker_key][node] &= ~mask

        timeout_us = float(getattr(self.cfg, "timeout_crash_us", 500000.0))
        if (self.env.now - job.arrival_time) * 1000.0 > timeout_us:
            job.state = JobState.KILLED
            if hasattr(self, "executor"): self.executor.num_killed += 1
            return
            
        if self.crashed or getattr(job, "state", None) == JobState.KILLED: return

        # Strict collision check against the ground truth (state.node_masks)
        if mask != 0 and (self.state.node_masks[node] & mask) == mask:
            job.demand_mask = mask  
            self.state.allocate(node, job.demand_mask)
            self.executor.start_job(job, node, True)
            return

        # Collision: resources snatched by a concurrent Broker, triggering retries
        region_key = (level, start, end)
        job.flux_tried_regions.add(region_key)
        parent_key = self._find_parent_key(level, start, end)
        if parent_key is None:
            self._handle_broker_fail(job, level, start, end)
            return

        hop_ms = float(getattr(self.cfg, "flux_hop_delay_us", 10.0)) / 1000.0
        self.env.schedule(hop_ms, self._enqueue_broker, *parent_key, job)