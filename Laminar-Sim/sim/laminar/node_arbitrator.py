import collections
from sim.core.job import JobState
from sim.mechanisms.two_phase_escrow import TwoPhaseEscrow

class NodeArbitrator:
    def __init__(self, env, state, executor, cfg) -> None:
        self.env = env
        self.state = state
        self.executor = executor
        self.cfg = cfg
        self.queues = collections.defaultdict(list)
        self.scheduled = set()
        self.two_phase = TwoPhaseEscrow(env, state, cfg)
        self.two_phase.arbitrator = self
        self.num_killed = 0

    def kill_probe(self, job) -> None:
        if job.state == JobState.KILLED: return
        job.state = JobState.KILLED
        self.state.metrics.control_work_us_array.append(job.control_work_us)
        self.executor.num_killed += 1
        self.num_killed += 1

    def receive_probe(self, job, node_id: int) -> None:
        if job.state == JobState.KILLED: return
        if hasattr(self, "dead_nodes") and node_id in self.dead_nodes: return 

        da_ttl_ms = float(getattr(self.cfg, "da_probe_ttl_ms", 50.0))
        da_start_attr = f"da_start_time_{getattr(job, 'probe_instance_id', 0)}"
        if hasattr(job, da_start_attr) and self.env.now - getattr(job, da_start_attr) > da_ttl_ms:
            self.kill_probe(job)
            return

        # DA probe enters local queue
        self.queues[node_id].append(job)
        
        # Core invariant: Heat (H) strictly equals queue depth
        # Real-time elevation of instantaneous queue heat for the node and its corresponding Zone
        self.state.node_h[node_id] += 1.0
        zid = self.state.zone_of_node(node_id)
        self.state.zone_h[zid] += 1.0

        if node_id not in self.scheduled:
            self.scheduled.add(node_id)
            self.env.schedule(getattr(self.cfg, "arb_window_ms", 1.0), self._run_window, node_id)

    def _run_window(self, node_id: int) -> None:
        self.scheduled.discard(node_id)
        q = self.queues[node_id]
        if not q:
            return

        window = self.queues[node_id]
        self.queues[node_id] = []
        
        # Physical law 1: Calculate absolute computation time for this batch of probes
        per_candidate_cost = getattr(self.cfg, "hw_mask_us", 0.004)
        batch_compute_ms = (len(window) * per_candidate_cost) / 1000.0

        for job in window:
            job.control_work_us += per_candidate_cost

        # Spatiotemporal isolation: Do not execute instantaneously!
        # Pause the node for batch_compute_ms, during which H (queue heat) broadcasts congestion warnings
        self.env.schedule(batch_compute_ms, self._finish_batch, node_id, window, batch_compute_ms)


    # Execution of settlement and business logic upon CPU computation completion
    def _finish_batch(self, node_id: int, window: list, compute_ms: float) -> None:
        
        # Release queue pressure
        # Strict mass conservation: released count must perfectly match inbound increments
        freed_h = float(len(window))
        self.state.node_h[node_id] = max(0.0, self.state.node_h[node_id] - freed_h)
        zid = self.state.zone_of_node(node_id)
        self.state.zone_h[zid] = max(0.0, self.state.zone_h[zid] - freed_h)

        # Physical law 2: Absolute Metronome pacing
        # E.g., if compute takes 0.1ms, rest 0.9ms. If 4ms, proceed immediately without resting
        base_window = getattr(self.cfg, "arb_window_ms", 1.0)
        next_delay_ms = max(0.0, base_window - compute_ms)

        if self.queues[node_id] and node_id not in self.scheduled:
            self.scheduled.add(node_id)
            self.env.schedule(next_delay_ms, self._run_window, node_id)

        # ==========================================================
        # Preserve all original arbitration business logic exactly as is:
        # ==========================================================
        # Ev_init acts as the local sorting weight
        candidates = [
            (getattr(j, "e_v_init", 1.0), j)
            for j in window
            if j.state != JobState.KILLED
        ]
        candidates.sort(key=lambda x: x[0], reverse=True)

        da_ttl_ms = float(getattr(self.cfg, "da_probe_ttl_ms", 50.0))

        for ev, candidate in candidates:
            if candidate.state == JobState.KILLED:
                continue

            # Defense Line 2: Re-check DA TTL upon dequeue to clear ghost probes caused by queuing backlogs
            da_start_attr = f"da_start_time_{getattr(candidate, 'probe_instance_id', 0)}"
            if hasattr(candidate, da_start_attr) and self.env.now - getattr(candidate, da_start_attr) > da_ttl_ms:
                self.kill_probe(candidate)
                continue

            is_contiguous = candidate.job_type in ["LLM_WHALE", "WHALE", "RED_SQUATTER"]
            free_mask = self.state.find_free_mask(node_id, int(candidate.mass), is_contiguous)

            if free_mask == 0:
                # False Optimism tracking: ZHAF projected feasibility, but local resources are exhausted
                if getattr(candidate, "expected_placable", False):
                    self.state.metrics.total_false_optimism_events += 1

                # Core fix: Bounce back to DA for backoff retry instead of direct kill
                if hasattr(self, "da_probe") and self.da_probe is not None:
                    self.da_probe.bounce(candidate, node_id)
                else:
                    self.kill_probe(candidate)
                continue

            # Resources available; proceed with standard Two-Phase / reservation flow (preserve original semantics)
            candidate.demand_mask = int(free_mask)
            if not self.two_phase.lock_and_deduct(candidate):
                self.kill_probe(candidate)
                continue

            if not self.state.allocate(node_id, candidate.demand_mask):
                self.kill_probe(candidate)
                self.env.crashed = True
                continue

            candidate.assigned_node = node_id
            candidate.state = JobState.RESERVED
            candidate.reservation_time = self.env.now
            self.state.metrics.total_reservations_granted += 1
            ttl_handle = self.two_phase.schedule_ttl(candidate, node_id)

            if not getattr(candidate, "is_malicious_squatter", False):
                if getattr(self.cfg, "enable_two_phase", True):
                    launch_ms = float(getattr(self.cfg, "network_rtt_ms", 0.5))
                    self.env.schedule(
                        launch_ms,
                        self._commit_reservation,
                        candidate,
                        node_id,
                        ttl_handle,
                    )
                else:
                    self.env.schedule(
                        0.0,
                        self._commit_reservation,
                        candidate,
                        node_id,
                        ttl_handle,
                    )

    def _commit_reservation(self, job, node_id: int, ttl_handle) -> None:
        if job.state != JobState.RESERVED: return
        if getattr(job, "pending_expired", False): return
        if getattr(job, "assigned_node", None) != node_id: return

        if ttl_handle is not None:
            try: ttl_handle.cancel()
            except Exception: pass

        self.state.metrics.control_work_us_array.append(job.control_work_us)
        self.executor.start_job(job, node_id, already_allocated=True)