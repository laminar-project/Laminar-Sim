from __future__ import annotations

def s(x: float) -> float:
    return float(x)

def ms(x: float) -> float:
    return float(x) / 1_000.0

def us(x: float) -> float:
    return float(x) / 1_000_000.0

def ns(x: float) -> float:
    return float(x) / 1_000_000_000.0
