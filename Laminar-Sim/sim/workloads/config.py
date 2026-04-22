from dataclasses import dataclass


@dataclass
class WorkloadConfig:
    """
    Dual-stream open-loop workload + optional red-team injection.

    Time unit convention:
    - *_arrival_rate_hz: jobs per second
    - max_time_ms: milliseconds
    - service times: milliseconds
    """

    # Throughput: two independent arrival streams
    sand_arrival_rate_hz: float = 2000.0
    whale_arrival_rate_hz: float = 200.0
    arrival_scale: float = 1.0
    max_time_ms: float = 1_800_000.0  # 30 minutes

    # Microservice ("sand"): exponential on ms
    sand_mean_ms: float = 5.0
    sand_slots: int = 4

    # Whale: lognormal on ms (mu/sigma in log-space)
    whale_mu: float = 4.0
    whale_sigma: float = 1.2
    whale_slots: int = 64

    # Energy model (Track II/III)
    base_e_v_init: float = 18.0
    base_e_patience: float = 18.0
    budget_multiplier: float = 1.0

    # Optional per-class economics multipliers
    sand_value_multiplier: float = 1.0
    whale_value_multiplier: float = 1.0
    sand_patience_multiplier: float = 1.0
    whale_patience_multiplier: float = 1.0
    sand_budget_multiplier: float = 1.0
    whale_budget_multiplier: float = 1.0

    # Red-team injection ratios
    squatter_ratio: float = 0.0
    liar_ratio: float = 0.0
    flood_ratio: float = 0.0
    liar_multiplier: float = 50.0
    hotspot_ratio: float = 0.0

    @property
    def total_arrival_rate_hz(self) -> float:
        return (self.sand_arrival_rate_hz + self.whale_arrival_rate_hz) * self.arrival_scale

    def effective_sand_rate_hz(self) -> float:
        return self.sand_arrival_rate_hz * self.arrival_scale

    def effective_whale_rate_hz(self) -> float:
        return self.whale_arrival_rate_hz * self.arrival_scale

    def __post_init__(self) -> None:
        if self.sand_arrival_rate_hz < 0:
            raise ValueError("sand_arrival_rate_hz must be >= 0")
        if self.whale_arrival_rate_hz < 0:
            raise ValueError("whale_arrival_rate_hz must be >= 0")
        if self.arrival_scale < 0:
            raise ValueError("arrival_scale must be >= 0")
        if self.max_time_ms <= 0:
            raise ValueError("max_time_ms must be > 0")
        if not (0.0 <= self.hotspot_ratio <= 1.0):
            raise ValueError("hotspot_ratio must be in [0, 1]")        

        if self.sand_mean_ms <= 0:
            raise ValueError("sand_mean_ms must be > 0")
        if self.whale_sigma < 0:
            raise ValueError("whale_sigma must be >= 0")

        for name, slots in [
            ("sand_slots", self.sand_slots),
            ("whale_slots", self.whale_slots),
        ]:
            if not (1 <= int(slots) <= 256):
                raise ValueError(f"{name} must be in [1, 256]")

        if self.base_e_v_init < 0:
            raise ValueError("base_e_v_init must be >= 0")
        if self.base_e_patience < 0:
            raise ValueError("base_e_patience must be >= 0")
        if self.budget_multiplier < 0:
            raise ValueError("budget_multiplier must be >= 0")

        for name, value in [
            ("sand_value_multiplier", self.sand_value_multiplier),
            ("whale_value_multiplier", self.whale_value_multiplier),
            ("sand_patience_multiplier", self.sand_patience_multiplier),
            ("whale_patience_multiplier", self.whale_patience_multiplier),
            ("sand_budget_multiplier", self.sand_budget_multiplier),
            ("whale_budget_multiplier", self.whale_budget_multiplier),
        ]:
            if value < 0:
                raise ValueError(f"{name} must be >= 0")

        if not (0.0 <= self.squatter_ratio <= 1.0):
            raise ValueError("squatter_ratio must be in [0, 1]")
        if not (0.0 <= self.liar_ratio <= 1.0):
            raise ValueError("liar_ratio must be in [0, 1]")
        if not (0.0 <= self.flood_ratio <= 1.0):
            raise ValueError("flood_ratio must be in [0, 1]")
        if self.squatter_ratio + self.liar_ratio + self.flood_ratio > 1.0 + 1e-12:
            raise ValueError("squatter_ratio + liar_ratio + flood_ratio must be <= 1.0")

        if self.liar_multiplier < 1.0:
            raise ValueError("liar_multiplier must be >= 1.0")
