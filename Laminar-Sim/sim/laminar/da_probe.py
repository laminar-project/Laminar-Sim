import math
import random
from sim.core.job import JobState


class DAProbe:
    def __init__(self, env, zhaf, config) -> None:
        self.env, self.zhaf, self.cfg = env, zhaf, config
        self._arbitrator = None
        self.num_killed = 0

    @property
    def arbitrator(self):
        return self._arbitrator

    @arbitrator.setter
    def arbitrator(self, val):
        self._arbitrator = val
        if val is not None:
            # Zero-intrusion reverse binding: allow arbitrator to invoke bounce()
            val.da_probe = self

    def max_candidates(self) -> int:
        return int(getattr(self.cfg, "da_k_probe", 16))

    def get_projected_node_view(self, candidates):
        return self.zhaf.get_projected_node_view(candidates)

    def _kill_probe(self, job) -> None:
        if job.state == JobState.KILLED:
            return
        # Prioritize arbitrator's kill_probe for consistent metrics
        if self.arbitrator is not None and hasattr(self.arbitrator, "kill_probe"):
            self.arbitrator.kill_probe(job)
        else:
            job.state = JobState.KILLED
            self.num_killed += 1

    # ===== Global constraints: TTL + Patience thresholds (independent of Two-Phase) =====

    def _check_da_lifetime(self, job) -> bool:
        """
        Returns True if the next action (scan or bounce) is allowed.
        Returns False if already killed.
        """
        # 1) Silence Window / TTL
        da_start_attr = f"da_start_time_{getattr(job, 'probe_instance_id', 0)}"
        if not hasattr(job, da_start_attr):
            # Record birth time upon first DA access
            setattr(job, da_start_attr, self.env.now)

        da_ttl_ms = float(getattr(self.cfg, "da_probe_ttl_ms", 50.0))
        if self.env.now - getattr(job, da_start_attr) > da_ttl_ms:
            self._kill_probe(job)
            return False

        # 2) Patience minimum action threshold (completely independent of Two-Phase escrow)
        action_cost = float(getattr(self.cfg, "da_action_cost", 5.0))
        bounce_cost = float(getattr(self.cfg, "da_bounce_cost", 10.0))
        default_min = min(action_cost, bounce_cost)
        min_next = float(getattr(self.cfg, "da_min_next_action_budget", default_min))

        if getattr(self.cfg, "enable_fast_fail", True) and job.e_patience < min_next:
            self._kill_probe(job)
            return False

        return True

    # ===== External interface: Attempt placement via launchpad node =====

    def send_to_arbitrator(self, job, best_node, total_delay_ms) -> None:
        if self.arbitrator is None:
            return
        self.env.schedule(total_delay_ms, self.arbitrator.receive_probe, job, best_node)

    def has_contiguous_ones(self, mask: int, length: int) -> bool:
        if length <= 0:
            return True
        target = (1 << length) - 1
        # Assume atomic single-node resource limit does not exceed 256
        for i in range(257 - length):
            if mask & (target << i) == (target << i):
                return True
        return False

    def select_candidates_in_zone(self, start_idx: int, end_idx: int):
        all_nodes = list(range(start_idx, end_idx))
        k = len(all_nodes)
        dak = self.max_candidates()
        if k <= dak:
            return all_nodes
        return random.sample(all_nodes, dak)

    def arrive_at_launchpad(self, job, current_node: int) -> None:
        """
        A standard placement attempt: sample and score within a zone via launchpad.
        """
        if job.state == JobState.KILLED:
            return

        # Constraint 1: TTL + Patience thresholds
        if not self._check_da_lifetime(job):
            return

        # Constraint 2: Deduct budget for this placement attempt
        action_cost = float(getattr(self.cfg, "da_action_cost", 5.0))
        job.e_patience -= action_cost

        # Record one control plane action
        self.zhaf.state.metrics.total_probe_actions += 1

        zone_id = self.zhaf.state.zone_of_node(current_node)
        start_idx = self.zhaf.state.zone_offsets[zone_id]
        end_idx = self.zhaf.state.zone_offsets[zone_id + 1]

        if end_idx <= start_idx:
            # Current zone is empty, trigger immediate bounce
            self.bounce(job, current_node)
            return

        candidates = self.select_candidates_in_zone(start_idx, end_idx)
        k = len(candidates)
        if k == 0:
            self.bounce(job, current_node)
            return

        dak_cost_us = float(getattr(self.cfg, "hw_da_compute_us", 0.45))
        job.control_work_us += dak_cost_us
        self.zhaf.state.metrics.total_node_score_evals += k

        S_arr, H_arr, A_arr = self.get_projected_node_view(candidates)
        best_node, best_score = -1, -float("inf")
        gamma = float(getattr(self.cfg, "teg_gamma", 1.0))
        sigma = float(getattr(self.cfg, "da_drift_sigma", 0.1))
        is_contiguous = job.job_type in ["LLM_WHALE", "WHALE", "RED_SQUATTER"]
        job.expected_placable = False

        for idx in range(k):
            sval = float(S_arr[idx])
            hval = float(H_arr[idx])
            Aj = int(A_arr[idx])

            if is_contiguous:
                placable = self.has_contiguous_ones(Aj, int(job.mass))
            else:
                placable = (Aj.bit_count() >= int(job.mass))

            if not placable and getattr(self.cfg, "enable_taylor", True):
                # Taylor approximation: consider placable if Scalar Slack is sufficient
                if sval >= float(job.mass):
                    placable = True

            if not placable:
                continue

            u_s = math.log2(1.0 + max(0.0, sval))
            u_h = math.log2(1.0 + max(0.0, hval))
            score = u_s - gamma * u_h + random.gauss(0.0, sigma)

            if score > best_score:
                best_score = score
                best_node = candidates[idx]
                job.expected_placable = True

        if best_node == -1:
            # No placable nodes found, backoff and retry
            self.bounce(job, current_node)
            return

        jitter_us = random.lognormvariate(
            math.log(max(1e-6, float(getattr(self.cfg, "os_jitter_mean_us", 50.0)))),
            float(getattr(self.cfg, "os_jitter_sigma", 1.2)),
        )
        total_delay_ms = (
            dak_cost_us / 1000.0
            + jitter_us / 1000.0
            + float(getattr(self.cfg, "network_rtt_ms", 0.5))
        )
        self.send_to_arbitrator(job, best_node, total_delay_ms)

    # ===== Core mechanism: Bounce backoff loop =====

    def bounce(self, job, current_node: int) -> None:
        """
        Trigger bounce retry upon False Optimism or empty zone instead of an immediate kill.
        """
        if job.state == JobState.KILLED:
            return

        # Constraint 1: TTL + Patience threshold (check eligibility before bounce)
        if not self._check_da_lifetime(job):
            return

        # Constraint 2: Deduct bounce penalty to accelerate exit under congestion/malicious scenarios
        bounce_cost = float(getattr(self.cfg, "da_bounce_cost", 10.0))
        job.e_patience -= bounce_cost

        # Randomly select a new launchpad within the zone, consuming physical RTT
        zone_id = self.zhaf.state.zone_of_node(current_node)
        start_idx = self.zhaf.state.zone_offsets[zone_id]
        end_idx = self.zhaf.state.zone_offsets[zone_id + 1]

        if end_idx > start_idx:
            next_node = random.randrange(start_idx, end_idx)
            delay = float(getattr(self.cfg, "network_rtt_ms", 0.5))
            
            # Physical injection: UDP packet loss during cross-node DA bounce
            loss_rate = float(getattr(self.cfg, "packet_loss_rate", 0.0))
            if random.random() >= loss_rate:
                self.env.schedule(delay, self.arrive_at_launchpad, job, next_node)
            else:
                # Probe physical annihilation (silent loss, does not invoke _kill_probe)
                # Await macro Regen mechanism to detect timeout and re-emit
                pass 
        else:
            # Zone degenerated to empty, physical truncation
            self._kill_probe(job)