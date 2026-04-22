from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class JobState(Enum):
    PENDING_PROBE = "PENDING_PROBE"
    RESERVED = "RESERVED"
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    KILLED = "KILLED"


@dataclass
class Job:
    job_id: str
    logical_task_id: str
    probe_instance_id: int
    job_type: str
    demand_mask: int
    mass: float
    base_duration_ms: float
    e_v_init: float
    e_patience: float

    state: JobState = JobState.PENDING_PROBE
    assigned_node: Optional[int] = None
    arrival_time: float = 0.0
    reservation_time: Optional[float] = None
    execution_start_time: Optional[float] = None
    finish_time: Optional[float] = None

    expected_placable: bool = False
    is_malicious_squatter: bool = False
    pending_expired: bool = False
    late_winner_reclaimed: bool = False
    harvested: bool = False

    control_work_us: float = 0.0

    finish_handle: Optional[object] = field(default=None, repr=False)
