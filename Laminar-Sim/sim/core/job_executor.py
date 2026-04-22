from __future__ import annotations
from typing import Dict, Optional

from sim.core.cluster_state import ClusterState
from sim.core.job import JobState

DEBUG_EXEC_PRINT: bool = False


class JobExecutor:
    def __init__(self, env, state: ClusterState) -> None:
        self.env = env
        self.state = state

        self.running_jobs: Dict[int, object] = {}
        self.num_completed: int = 0
        self.num_killed: int = 0

    def start_job(self, job, node_id: int, already_allocated: bool = False) -> None:
        logical_task_id = getattr(job, "logical_task_id", None)

        if logical_task_id is not None and logical_task_id in self.state.started_logical_tasks:
            self.state.metrics.total_uniqueness_violations += 1

            if already_allocated:
                demand = int(getattr(job, "demand_mask", 0))
                if demand != 0:
                    self.state.release(node_id, demand)

                if getattr(job, "reservation_time", None) is not None:
                    self.state.metrics.late_winner_residence_times.append(
                        self.env.now - job.reservation_time
                    )

                self.state.metrics.total_late_winner_reclaims += 1
                job.late_winner_reclaimed = True

            job.state = JobState.KILLED
            job.assigned_node = None
            self.num_killed += 1
            return

        job.state = JobState.STARTED
        job.assigned_node = node_id
        job.execution_start_time = self.env.now

        if getattr(job, "arrival_time", None) is not None:
            self.state.metrics.start_lats_all.append(
                job.execution_start_time - job.arrival_time
            )

        self.state.metrics.total_execution_starts += 1
        self.state.metrics.total_successful_starts += 1

        if logical_task_id is not None:
            self.state.started_logical_tasks.add(logical_task_id)

        if not getattr(job, "is_malicious_squatter", False):
            self.state.metrics.total_honest_goodput += 1

        if not already_allocated:
            demand = int(getattr(job, "demand_mask", 0))
            if demand != 0:
                self.state.allocate(node_id, demand)

        self.running_jobs[id(job)] = job

        run_ms = float(getattr(job, "service_time_ms", getattr(job, "base_duration_ms", 1.0)))
        self.env.schedule(run_ms, self._finish_job, job, node_id)

    def _finish_job(self, job, node_id: int) -> None:
        if getattr(job, "state", None) == JobState.KILLED:
            self.running_jobs.pop(id(job), None)
            return

        job.finish_time = self.env.now
        job.state = JobState.COMPLETED

        exec_ms = 0.0
        if getattr(job, "execution_start_time", None) is not None:
            exec_ms = job.finish_time - job.execution_start_time

        if DEBUG_EXEC_PRINT:
            print(
                f"[EXEC] job_id={getattr(job, 'job_id', id(job))} "
                f"exec_ms={exec_ms:.3f}"
            )

        demand = int(getattr(job, "demand_mask", 0))
        if demand != 0:
            self.state.release(node_id, demand)

        self.num_completed += 1
        self.state.metrics.total_completions += 1

        self.running_jobs.pop(id(job), None)

    def kill_job(self, job, reason: Optional[str] = None) -> None:
        if getattr(job, "state", None) in (JobState.COMPLETED, JobState.KILLED):
            return

        old_state = getattr(job, "state", None)
        node_id = getattr(job, "assigned_node", None)

        job.state = JobState.KILLED
        job.kill_reason = reason

        demand = int(getattr(job, "demand_mask", 0))

        if id(job) in self.running_jobs:
            if node_id is not None and demand != 0:
                self.state.release(node_id, demand)
            self.running_jobs.pop(id(job), None)
        elif old_state == JobState.RESERVED:
            if node_id is not None and demand != 0:
                self.state.release(node_id, demand)
            job.assigned_node = None

        self.num_killed += 1
