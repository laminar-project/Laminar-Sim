import copy
import random
import hashlib

from sim.core import Job
from sim.core.job import JobState
from sim.laminar.config import LaminarConfig

class WorkloadGenerator:
    def __init__(self, env, config, state, submit_callback, rng=None) -> None:
        self.env, self.cfg, self.state, self.submit, self.rng = (
            env, config, state, submit_callback, rng or random.Random()
        )
        self.task_counter = 0
        self.sand_counter = 0
        self.whale_counter = 0

        lam_cfg = LaminarConfig()
        self.escrow_fee = float(getattr(self.cfg, "escrow_fee", getattr(lam_cfg, "two_phase_escrow", 50.0)))
        self.enable_two_phase = bool(getattr(self.cfg, "enable_two_phase", getattr(lam_cfg, "enable_two_phase", True)))

    def start(self) -> None:
        self.schedule_next_sand()
        self.schedule_next_whale()

    def schedule_next_sand(self) -> None:
        if self.env.now >= self.cfg.max_time_ms: return
        rate = float(self.cfg.sand_arrival_rate_hz) / 1000.0
        if rate > 0.0: self.env.schedule(max(0.001, self.rng.expovariate(rate)), self.fire_sand)

    def schedule_next_whale(self) -> None:
        if self.env.now >= self.cfg.max_time_ms: return
        rate = float(self.cfg.whale_arrival_rate_hz) / 1000.0
        if rate > 0.0: self.env.schedule(max(0.001, self.rng.expovariate(rate)), self.fire_whale)

    def fire_sand(self) -> None:
        if self.env.now >= self.cfg.max_time_ms: return
        self.fire_logical_task(False)
        self.schedule_next_sand()

    def fire_whale(self) -> None:
        if self.env.now >= self.cfg.max_time_ms: return
        self.fire_logical_task(True)
        self.schedule_next_whale()

    def fire_logical_task(self, is_whale: bool) -> None:
        self.task_counter += 1
        self.state.metrics.total_arrivals += 1
        logical_id = f"T{self.task_counter}"

        job_type = "LLM_WHALE" if is_whale else "MICROSERVICE"
        dur = max(0.1, self.rng.lognormvariate(float(self.cfg.whale_mu), float(self.cfg.whale_sigma))) if is_whale else max(0.1, self.rng.expovariate(1.0 / float(self.cfg.sand_mean_ms)))
        slots = int(self.cfg.whale_slots) if is_whale else int(self.cfg.sand_slots)

        if is_whale:
            start = self.rng.randint(0, 256 - slots)
            mask = ((1 << slots) - 1) << start
        else:
            positions = self.rng.sample(range(256), slots)
            mask = 0
            for pos in positions: mask |= (1 << pos)

        mass = float(slots)
        ev = float(self.cfg.base_e_v_init) * mass
        epat = float(self.cfg.base_e_patience) * mass * float(self.cfg.budget_multiplier)

        is_sq = self.rng.random() < float(getattr(self.cfg, "squatter_ratio", 0.0))
        if is_sq:
            job_type, mask, mass, ev, epat = "RED_SQUATTER", ((1 << 64) - 1) << 1, 64.0, 1e9, 1e9

        job = Job(
            job_id=f"{logical_id}_p0", logical_task_id=logical_id, probe_instance_id=0,
            job_type=job_type, demand_mask=mask, mass=mass, base_duration_ms=dur,
            e_v_init=ev, e_patience=epat,
        )
        job.arrival_time = float(self.env.now)
        if is_sq: job.is_malicious_squatter = True
        hotspot_ratio = float(getattr(self.cfg, "hotspot_ratio", 0.0))
        if hotspot_ratio > 0.0:
            h = int(hashlib.md5(logical_id.encode("utf-8")).hexdigest(), 16)
            task_rng = random.Random(h)
            if task_rng.random() < hotspot_ratio:
                hot_nodes = max(1, self.state.num_nodes // 20)   # top 5% nodes
                job.entry_node = task_rng.randrange(hot_nodes)
            else:
                job.entry_node = task_rng.randrange(self.state.num_nodes)
        else:
            job.entry_node = self.rng.randrange(self.state.num_nodes)
        self.submit(job)

        if getattr(self.cfg, "enable_regeneration", True) and not is_sq:
            # Restore macro-level probe fallback cycle.
            # Allow the network sufficient time for thermodynamic dissipation instead of aggressively injecting new probes.
            regen_ms = float(getattr(self.cfg, "regen_timeout_ms", 150.0))
            self.env.schedule(regen_ms, self.check_regen, job)

    def check_regen(self, orig) -> None:
        if self.env.now >= self.cfg.max_time_ms: 
            return
        if orig.logical_task_id in self.state.started_logical_tasks: 
            return

        # =====================================================================
        # Physical Law: Absolute client-side SLA lifetime threshold.
        # Strict fairness baseline: Laminar is also subject to the strict SLA timeout boundary. 
        # Tasks exceeding this are aborted immediately.
        # =====================================================================
        # Extract globally unified SLA timeout threshold (with configuration fallback)
        timeout_us = 500_000.0  # Force a 500ms physical deadline
        if hasattr(self.cfg, "timeout_crash_us"):
            timeout_us = float(self.cfg.timeout_crash_us)
            
        if (self.env.now - orig.arrival_time) * 1000.0 > timeout_us:
            orig.state = JobState.KILLED
            if hasattr(self, "executor"):
                self.executor.num_killed += 1# Task physically disconnected by client; cease DA regeneration and release memory
            return 
        # =====================================================================

        # Correctly increment generation count on the original object to prevent infinite cloning loops.
        orig.probe_instance_id += 1
        
        # Cap maximum macroscopic regenerations at 5 (typically preempted by TTL prior to this)
        if orig.probe_instance_id >= 5: 
            return
            
        if getattr(orig, "e_patience", 0.0) <= 0.0: 
            return
            
        if getattr(self.cfg, "enable_two_phase", True) and getattr(orig, "e_patience", 0.0) < getattr(self.cfg, "escrow_fee", 50.0): 
            return

        self.state.metrics.total_duplicate_das += 1

        # Generate a new DA probe replica
        import copy
        njob = copy.copy(orig)
        njob.job_id = f"{orig.logical_task_id}_p{njob.probe_instance_id}"

        # Ensure correct invocation of the externally provided submit callback
        if hasattr(self, "submit") and self.submit:
            self.submit(njob)

        # Recursive scheduling must adhere to the macroscopic cycle (e.g., 150ms) rather than a hardcoded rapid interval
        regen_ms = float(getattr(self.cfg, "regen_timeout_ms", 150.0))
        self.env.schedule(regen_ms, self.check_regen, orig)