from dataclasses import dataclass


@dataclass
class LaminarConfig:
    enable_taylor: bool = True
    enable_missingness_guard: bool = True
    enable_two_phase: bool = True
    enable_fast_fail: bool = True

    hw_fma_us_per_zone: float = 0.02929
    hw_da_compute_us: float = 0.0137
    hw_mask_us: float = 0.00402

    teg_gamma: float = 1.0

    zhaf_broadcast_ms: float = 10.0
    network_rtt_ms: float = 0.1
    packet_loss_rate: float = 0.01
    da_drift_sigma: float = 0.5

    arb_window_ms: float = 1.0
    two_phase_ttl_ms: float = 200.0
    two_phase_escrow: float = 20.0
    # === DA Addressing Mechanism Constraints (Independent of Two-Phase Escrow) ===
    da_probe_ttl_ms: float = 150.0      # Absolute DA silence window/TTL
    da_action_cost: float = 3.0        # Patience budget consumed per scanning/addressing action
    da_bounce_cost: float = 6.0       # Additional patience penalty per bounce/retry
    da_min_next_action_budget: float = 1.0  # Minimum patience threshold to trigger Fast-Fail for the next action