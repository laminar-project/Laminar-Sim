from sim.core.job import JobState

class TwoPhaseEscrow:
    def __init__(self, env, state, config):
        self.env = env
        self.state = state
        self.cfg = config
        self.arbitrator = None

    def lock_and_deduct(self, job) -> bool:
        if not getattr(self.cfg, "enable_two_phase", True):
            return True
        escrow_fee = getattr(self.cfg, "two_phase_escrow", 50.0)
        if job.e_patience >= escrow_fee:
            job.e_patience -= escrow_fee
            return True
        return False

    def schedule_ttl(self, job, node_id: int):
        if not getattr(self.cfg, "enable_two_phase", True):
            return None
        ttl_ms = getattr(self.cfg, "two_phase_ttl_ms", 500.0)
        return self.env.schedule(ttl_ms, self._execute_timeout, job, node_id)

    def _execute_timeout(self, job, node_id: int):
        if job.state == JobState.RESERVED and job.assigned_node == node_id:
            job.pending_expired = True
            self.state.metrics.total_pending_expiry += 1

            if job.reservation_time is not None:
                self.state.metrics.reservation_hold_times.append(self.env.now - job.reservation_time)

            demand = int(getattr(job, "demand_mask", 0))
            if demand != 0:
                self.state.release(node_id, demand)

            job.assigned_node = None
            job.state = JobState.KILLED
            if self.arbitrator is not None:
                self.arbitrator.executor.num_killed += 1
                self.arbitrator.num_killed += 1

    def cancel_ttl(self, ttl_handle):
        if getattr(self.cfg, "enable_two_phase", True) and ttl_handle:
            ttl_handle.cancel()