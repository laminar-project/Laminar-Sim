# sim/baselines/centralized.py

import math
from collections import deque
from typing import Optional

from sim.core import EventLoop, ClusterState, JobExecutor
from sim.core.job import JobState
from .config import BaselineConfig

class SlurmBaseline:
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

        self.metrics_decision_us = []
        self.crashed = False

        self.queue = deque()
        self.is_busy = False

    def submit_job(self, job) -> None:
        if self.crashed:
            return

        max_q = int(getattr(self.cfg, "slurm_max_queue", 200000))
        if len(self.queue) >= max_q:
            job.state = JobState.KILLED
            self.executor.num_killed += 1
            return

        if getattr(job, "retries", None) is None:
            job.retries = 0

        self.queue.append(job)
        if not self.is_busy:
            self.is_busy = True
            self.env.schedule(0.0, self._process_next)

    def submit(self, job) -> None:
        self.submit_job(job)

    def _maybe_crash(self, decision_us: float) -> bool:
        if not self.enable_crash:
            return False

        timeout_us = float(getattr(self.cfg, "timeout_crash_us", 1_000_000.0))
        if decision_us <= timeout_us:
            return False

        self.crashed = True
        while self.queue:
            j = self.queue.popleft()
            if getattr(j, "state", None) not in (JobState.COMPLETED, JobState.KILLED):
                j.state = JobState.KILLED
                self.executor.num_killed += 1
        self.is_busy = False
        return True

    def _find_free_node(self, job) -> Optional[int]:
        is_contig = getattr(job, "job_type", None) in ("LLM_WHALE", "WHALE", "RED_SQUATTER")
        job_mass = float(job.mass)

        for nid in range(self.state.num_nodes):
            if self.enable_scalar_filter and getattr(self.state, "node_s", None) is not None:
                if float(self.state.node_s[nid]) < job_mass:
                    continue

            free_mask = self.state.find_free_mask_from_bits(
                int(self.state.node_masks[nid]),
                int(job_mass),
                is_contig,
            )
            if free_mask != 0:
                job.demand_mask = int(free_mask)
                return nid

        return None

    def _process_next(self) -> None:
        if self.crashed or not self.queue:
            self.is_busy = False
            return

        job = self.queue[0]
        q_len = len(self.queue)

        alpha_base = float(getattr(self.cfg, "slurm_base_scan_us", 1.0))
        beta_node = float(getattr(self.cfg, "slurm_per_node_us", 0.0001))
        gamma_lock = float(getattr(self.cfg, "slurm_lock_base_us", 0.5))
        s_scale = float(getattr(self.cfg, "slurm_lock_scale", 10000.0))

        t_scan = alpha_base + beta_node * self.state.num_nodes
        t_lock = gamma_lock * (1.0 + q_len / max(1.0, s_scale))
        decision_us = t_scan + t_lock

        self.metrics_decision_us.append(float(decision_us))
        if self._maybe_crash(decision_us):
            return

        self.env.schedule(decision_us / 1000.0, self._finish_placement, job)

    def _delayed_requeue(self, job) -> None:
        if self.crashed:
            return
        self.queue.append(job)
        if not self.is_busy:
            self.is_busy = True
            self.env.schedule(0.0, self._process_next)

    def _finish_placement(self, job) -> None:
        if self.crashed:
            self.is_busy = False
            return

        if not self.queue or self.queue[0] is not job:
            if self.queue:
                self.env.schedule(0.0, self._process_next)
            else:
                self.is_busy = False
            return

        node = self._find_free_node(job)

        if node is not None:
            self.queue.popleft()
            self.state.allocate(node, job.demand_mask)
            job.assigned_node = node
            self.executor.start_job(job, node, already_allocated=True)
            if self.queue:
                self.env.schedule(0.0, self._process_next)
            else:
                self.is_busy = False
            return

        max_retries = int(getattr(self.cfg, "slurm_max_retries", 5))
        base_ms = float(getattr(self.cfg, "slurm_retry_base_ms", 2.0))

        if getattr(job, "retries", 0) < max_retries:
            job.retries += 1
            self.queue.popleft()
            backoff_ms = base_ms * (2 ** min(job.retries, 8))
            self.env.schedule(backoff_ms, self._delayed_requeue, job)

            if self.queue:
                self.env.schedule(0.0, self._process_next)
            else:
                self.is_busy = False
            return

        self.queue.popleft()
        job.state = JobState.KILLED
        self.executor.num_killed += 1
        if self.queue:
            self.env.schedule(0.0, self._process_next)
        else:
            self.is_busy = False