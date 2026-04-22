# sim/run_exp.py

import math
import hashlib
import random

from sim.core import EventLoop, ClusterState, JobExecutor
from sim.workloads import WorkloadConfig, WorkloadGenerator
from sim.laminar import LaminarConfig, ZHAFMesh, TEGGateway, DAProbe, NodeArbitrator
from sim.baselines import BaselineConfig, SlurmBaseline, RayBaseline, FluxBaseline
from sim.mechanisms.zone_layout import generate_hetero_zone_sizes


def calc_p(arr, pct: float = 0.99) -> float:
    if not arr:
        return float("nan")
    arr_sorted = sorted(arr)
    idx = min(int(len(arr_sorted) * pct), len(arr_sorted) - 1)
    return float(arr_sorted[idx])


def calc_stats(arr):
    if not arr:
        return {
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "mean": float("nan"),
            "count": 0,
        }

    xs = sorted(float(x) for x in arr)
    n = len(xs)

    def q(p):
        idx = min(int(n * p), n - 1)
        return float(xs[idx])

    return {
        "p50": q(0.50),
        "p95": q(0.95),
        "p99": q(0.99),
        "mean": float(sum(xs) / n),
        "count": n,
    }


def run_single(exp_name: str, params: dict) -> dict:
    seed_val = int(hashlib.md5(exp_name.encode("utf-8")).hexdigest(), 16) % (2**32)
    random.seed(seed_val)

    scheduler_name = params.get("scheduler", "laminar")
    nodes = int(params.get("nodes", 1000))
    rho = float(params.get("rho", 0.5))
    loss = float(params.get("loss", 0.0))
    max_time_ms = float(params.get("max_time_ms", 30000.0))

    # ---- Explicit stress knobs for Exp2 / realistic hierarchical baselines ----
    hotspot_ratio = float(params.get("hotspot_ratio", 0.0))
    ray_hotspot_scale = float(params.get("ray_hotspot_scale", 500.0))
    flux_root_scale = float(params.get("flux_root_scale", 1500.0))

    target_hz = (
        (rho * nodes * 256.0)
        / (0.8 * 20.0 + 0.2 * (64.0 * math.exp(4.0 + (1.2 ** 2) / 2.0)))
        * 1000.0
    )

    wc = WorkloadConfig(
        sand_arrival_rate_hz=target_hz * 0.8,
        whale_arrival_rate_hz=target_hz * 0.2,
        max_time_ms=max_time_ms,
        squatter_ratio=params.get("squatter_ratio", 0.0),
    )

    # Explicit workload-side skew knob; safe whether or not dataclass already has the field
    setattr(wc, "hotspot_ratio", hotspot_ratio)

    setattr(
        wc,
        "enable_regeneration",
        params.get("enable_regeneration", True) if scheduler_name == "laminar" else False,
    )

    lam_cfg = LaminarConfig()
    lam_cfg.enable_taylor = params.get("enable_taylor", True)
    lam_cfg.enable_missingness_guard = params.get("enable_missingness_guard", True)
    lam_cfg.enable_two_phase = params.get("enable_two_phase", True)
    lam_cfg.packet_loss_rate = loss if scheduler_name == "laminar" else 0.0

    # Optional Laminar mechanism overrides
    if "two_phase_ttl_ms" in params:
        lam_cfg.two_phase_ttl_ms = float(params["two_phase_ttl_ms"])
    if "two_phase_escrow" in params:
        lam_cfg.two_phase_escrow = float(params["two_phase_escrow"])
    if "arb_window_ms" in params:
        lam_cfg.arb_window_ms = float(params["arb_window_ms"])
    # Connect parameters for Exp3 ablation study
    if "da_drift_sigma" in params:
        lam_cfg.da_drift_sigma = float(params["da_drift_sigma"])
    if "teg_gamma" in params:
        lam_cfg.teg_gamma = float(params["teg_gamma"])

    setattr(wc, "enable_two_phase", getattr(lam_cfg, "enable_two_phase", True))
    setattr(wc, "escrow_fee", float(getattr(lam_cfg, "two_phase_escrow", 50.0)))

    base_cfg = BaselineConfig()

    def _pick(*keys, default=None):
        for k in keys:
            if k in params and params[k] is not None:
                return params[k]
        return default

    def _as_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

    # ------------------------------------------------------------
    # 0) Global baseline knobs
    # ------------------------------------------------------------
    v = _pick("timeout_crash_us")
    if v is not None:
        base_cfg.timeout_crash_us = float(v)

    v = _pick("enable_crash")
    if v is not None:
        base_cfg.enable_crash = _as_bool(v)

    v = _pick("enable_scalar_filter")
    if v is not None:
        base_cfg.enable_scalar_filter = _as_bool(v)

    v = _pick("baseline_heartbeat_ms")
    if v is not None:
        base_cfg.baseline_heartbeat_ms = float(v)

    # ------------------------------------------------------------
    # 1) Slurm knobs
    # ------------------------------------------------------------
    v = _pick("slurm_base_scan_us")
    if v is not None:
        base_cfg.slurm_base_scan_us = float(v)

    v = _pick("slurm_per_node_us")
    if v is not None:
        base_cfg.slurm_per_node_us = float(v)

    v = _pick("slurm_lock_base_us")
    if v is not None:
        base_cfg.slurm_lock_base_us = float(v)

    v = _pick("slurm_lock_scale")
    if v is not None:
        base_cfg.slurm_lock_scale = float(v)

    v = _pick("slurm_max_retries")
    if v is not None:
        base_cfg.slurm_max_retries = int(v)

    v = _pick("slurm_retry_base_ms")
    if v is not None:
        base_cfg.slurm_retry_base_ms = float(v)

    v = _pick("slurm_max_queue")
    if v is not None:
        base_cfg.slurm_max_queue = int(v)

    # ------------------------------------------------------------
    # 2) Ray knobs
    # ------------------------------------------------------------
    v = _pick("ray_local_k")
    if v is not None:
        base_cfg.ray_local_k = int(v)

    v = _pick("ray_local_base_us")
    if v is not None:
        base_cfg.ray_local_base_us = float(v)

    v = _pick("ray_gcs_base_us")
    if v is not None:
        base_cfg.ray_gcs_base_us = float(v)

    v = _pick("ray_gcs_shards")
    if v is not None:
        base_cfg.ray_gcs_shards = int(v)

    v = _pick("ray_gcs_hotspot_scale")
    if v is not None:
        base_cfg.ray_gcs_hotspot_scale = float(v)
    else:
        base_cfg.ray_gcs_hotspot_scale = float(ray_hotspot_scale)

    v = _pick("ray_gcs_shard_bias")
    if v is not None:
        base_cfg.ray_gcs_shard_bias = float(v)

    v = _pick("ray_max_spillbacks")
    if v is not None:
        base_cfg.ray_max_spillbacks = int(v)

    v = _pick("ray_retry_base_ms")
    if v is not None:
        base_cfg.ray_retry_base_ms = float(v)

    v = _pick("ray_network_rtt_us")
    if v is not None:
        base_cfg.ray_network_rtt_us = float(v)

    v = _pick("ray_gcs_max_queue")
    if v is not None:
        base_cfg.ray_gcs_max_queue = int(v)

    # ------------------------------------------------------------
    # 3) Flux knobs
    # ------------------------------------------------------------
    v = _pick("flux_tree_fanout")
    if v is not None:
        base_cfg.flux_tree_fanout = int(v)

    v = _pick("flux_match_base_us")
    if v is not None:
        base_cfg.flux_match_base_us = float(v)

    v = _pick("flux_per_node_us")
    if v is not None:
        base_cfg.flux_per_node_us = float(v)

    v = _pick("flux_hop_delay_us")
    if v is not None:
        base_cfg.flux_hop_delay_us = float(v)

    v = _pick("flux_root_scale")
    if v is not None:
        base_cfg.flux_root_scale = float(v)
    else:
        base_cfg.flux_root_scale = float(flux_root_scale)

    v = _pick("flux_max_retries")
    if v is not None:
        base_cfg.flux_max_retries = int(v)

    v = _pick("flux_retry_base_ms")
    if v is not None:
        base_cfg.flux_retry_base_ms = float(v)

    v = _pick("flux_leaf_k")
    if v is not None:
        base_cfg.flux_leaf_k = int(v)

    v = _pick("flux_root_max_queue")
    if v is not None:
        base_cfg.flux_root_max_queue = int(v)

    # ------------------------------------------------------------
    # 4) Optional aliases for old experiment scripts
    #    These aliases preserve backward compatibility for legacy configuration keys.
    # ------------------------------------------------------------
    alias_map = {
        "raylocalk": ("ray_local_k", int),
        "raylocalbaseus": ("ray_local_base_us", float),
        "raygcsbaseus": ("ray_gcs_base_us", float),
        "raygcsshards": ("ray_gcs_shards", int),
        "raygcshotspotscale": ("ray_gcs_hotspot_scale", float),
        "raygcsshardbias": ("ray_gcs_shard_bias", float),
        "raymaxspillbacks": ("ray_max_spillbacks", int),
        "rayretrybasems": ("ray_retry_base_ms", float),
        "raynetworkrttus": ("ray_network_rtt_us", float),
        "timeoutcrashus": ("timeout_crash_us", float),
        "fluxtreefanout": ("flux_tree_fanout", int),
        "fluxmatchbaseus": ("flux_match_base_us", float),
        "fluxpernodeus": ("flux_per_node_us", float),
        "fluxhopdelayus": ("flux_hop_delay_us", float),
        "fluxrootscale": ("flux_root_scale", float),
        "fluxmaxretries": ("flux_max_retries", int),
        "fluxretrybasems": ("flux_retry_base_ms", float),
        "fluxleafk": ("flux_leaf_k", int),
    }

    for old_key, (new_attr, caster) in alias_map.items():
        if old_key in params and params[old_key] is not None:
            setattr(base_cfg, new_attr, caster(params[old_key]))

    # Handle the amplification effect of packet loss in the macro environment
    if loss > 0.0:
        lm = 1.0 / (1.0 - loss)
        base_cfg.ray_network_rtt_us *= lm
        base_cfg.flux_hop_delay_us *= lm
        base_cfg.slurm_base_scan_us *= lm
        base_cfg.slurm_lock_base_us *= lm
        if scheduler_name == "laminar":
            lam_cfg.network_rtt_ms *= lm

    hetero_zones = params.get("hetero_zones", False)

    if hetero_zones and scheduler_name == "laminar":
        zone_sizes = generate_hetero_zone_sizes(
            total_nodes=nodes,
            target_zone_size=params.get("target_zone_size", 1024),
            jitter=params.get("zone_jitter", 0.20),
            seed=seed_val,
        )
        env = EventLoop()
        cluster = ClusterState(num_nodes=nodes, zone_sizes=zone_sizes)
    else:
        env = EventLoop()
        cluster = ClusterState(num_nodes=nodes)

    executor = JobExecutor(env, cluster)

    if scheduler_name == "laminar":
        zhaf = ZHAFMesh(env, cluster, lam_cfg)
        scheduler = TEGGateway(env, cluster, lam_cfg, zhaf)
        da = DAProbe(env, zhaf, lam_cfg)
        arbitrator = NodeArbitrator(env, cluster, executor, lam_cfg)
        scheduler.da_probe = da
        da.arbitrator = arbitrator
        metrics_source = cluster.metrics
    else:
        scheduler = (
            SlurmBaseline(env, cluster, executor, base_cfg)
            if scheduler_name == "slurm"
            else RayBaseline(env, cluster, executor, base_cfg)
            if scheduler_name == "ray"
            else FluxBaseline(env, cluster, executor, base_cfg)
        )
        metrics_source = scheduler

    submit_fn = getattr(scheduler, "submit_job", getattr(scheduler, "submit", None))

    gen = WorkloadGenerator(
        env,
        wc,
        cluster,
        submit_callback=submit_fn,
        rng=random.Random(seed_val),
    )
    gen.start()

    # =========================================================================
    # Physical observation and chaos injection
    # =========================================================================
    timeseries_log = []
    last_exec_starts = 0

    def heartbeat():
        nonlocal last_exec_starts
        current_sim_sec = env.now / 1000.0
        total_sim_sec = max_time_ms / 1000.0
        pct = (current_sim_sec / total_sim_sec) * 100.0

        # Calculate per-second timeseries data (for Exp 4 timeline plots)
        current_starts = cluster.metrics.total_execution_starts
        throughput_per_sec = current_starts - last_exec_starts
        last_exec_starts = current_starts

        # Physical utilization = (Total Slots - Idle Slots) / Total Slots
        used_capacity = (nodes * 256.0) - sum(cluster.node_s)
        utilization = used_capacity / (nodes * 256.0) if nodes > 0 else 0.0

        timeseries_log.append({
            "time_sec": current_sim_sec,
            "throughput": throughput_per_sec,
            "utilization": utilization
        })
        # --------------------------------------------------------

        # Extract absolute arrival and start counts
        arrivals = cluster.metrics.total_arrivals
        starts = cluster.metrics.total_execution_starts

        # Academic correction: Evaluate control-plane survivability directly via instantaneous Success Ratio
        success_ratio = (starts / arrivals * 100.0) if arrivals > 0 else 100.0

        # Retain utilization calculation for timeseries plotting only
        used_capacity = (nodes * 256.0) - sum(cluster.node_s)
        utilization = used_capacity / (nodes * 256.0) if nodes > 0 else 0.0

        timeseries_log.append({
            "time_sec": current_sim_sec,
            "throughput": throughput_per_sec,
            "utilization": utilization
        })

        # Append SuccRatio to terminal output
        print(f"⏳ [{scheduler_name.upper()} | {exp_name}] "
              f"Time: {current_sim_sec:.1f}s / {total_sim_sec:.1f}s ({pct:.1f}%) "
              f"| Arrivals: {arrivals} | Starts: {starts} "
              f"| SuccRatio: {success_ratio:.1f}%", flush=True)

        if env.now + 1000.0 < max_time_ms:
            env.schedule(1000.0, heartbeat)

    env.schedule(1000.0, heartbeat)

    # Chaos Monkey: deterministic physical node failure injection
    chaos_kill_ratio = float(params.get("chaos_kill_ratio", 0.0))
    if scheduler_name == "laminar" and chaos_kill_ratio > 0.0:
        def inject_node_silence():
            kill_count = int(nodes * chaos_kill_ratio)
            # Oracle selection of physical nodes to be silenced
            dead_nodes = set(random.sample(range(nodes), kill_count))
            # Signal ZHAF that these nodes are physically partitioned
            if hasattr(zhaf, "dead_nodes"):
                zhaf.dead_nodes = dead_nodes
            if hasattr(arbitrator, "dead_nodes"):
                arbitrator.dead_nodes = dead_nodes
            print(f"\n💥 [CHAOS] {kill_count} nodes physically silenced at T={env.now/1000.0}s!\n")

        # Trigger disaster at 30% experiment progress
        env.schedule(max_time_ms * 0.3, inject_node_silence)
    # =========================================================================
    # =========================================================================

    env.run(max_time_ms)

    m = cluster.metrics
    is_crashed = getattr(scheduler, "crashed", False) or getattr(env, "crashed", False)

    arrivals = max(m.total_arrivals, 1)
    reservations = max(m.total_reservations_granted, 1)
    exec_starts = max(m.total_execution_starts, 1)

    if scheduler_name == "laminar":
        p99_dec = calc_p(m.control_work_us_array, 0.99)
        if m.control_work_us_array:
            cw_per_success = sum(m.control_work_us_array) / exec_starts
        else:
            cw_per_success = 0.0

        local_stats = calc_stats([])
        gcs_service_stats = calc_stats([])
        gcs_wait_stats = calc_stats([])

    else:
        # Metric alignment: Eliminate sample dilution
        # Extract independent service times for the global central control plane (Ray GCS or Flux Root)
        gcs_svc = getattr(metrics_source, "metrics_gcs_service_us", [])
        root_svc = getattr(metrics_source, "metrics_root_service_us", [])
        global_svc_samples = gcs_svc if gcs_svc else root_svc

        # p99_dec specifically reflects the single-dispatch latency of the central control plane
        # Under high pressure, this demonstrates the thrashing degradation modeled by the Universal Scalability Law (USL)
        p99_dec = calc_p(global_svc_samples, 0.99)

        # Calculate total control-plane compute cost (Local/Leaf fast-path + Central retry requests)
        local_svc = getattr(metrics_source, "metrics_local_service_us", [])
        leaf_svc = getattr(metrics_source, "metrics_leaf_service_us", [])
        all_local_svc = local_svc if local_svc else leaf_svc

        gcs_wait = getattr(metrics_source, "metrics_gcs_wait_us", [])

        total_svc_sum = sum(global_svc_samples) + sum(all_local_svc)
        cw_per_success = (total_svc_sum / exec_starts) if exec_starts > 0 else 0.0

        local_stats = calc_stats(all_local_svc)
        gcs_service_stats = calc_stats(global_svc_samples)
        gcs_wait_stats = calc_stats(gcs_wait)

    # -------------------------------------------------------------
    # SLA Death Penalty Enforcement
    # -------------------------------------------------------------
    # Extract SLA timeout ceiling and convert to milliseconds (default 500.0 ms)
    timeout_ms = getattr(base_cfg, "timeout_crash_us", 500000.0) / 1000.0

    if is_crashed:
        # System hard crash: penalize all tasks with maximum SLA timeout
        p99_start = timeout_ms
        p99_dec = getattr(base_cfg, "timeout_crash_us", 500000.0)
    else:
        # 1. Extract true queuing latencies of all surviving tasks
        true_lats = list(cluster.metrics.start_lats_all)

        # 2. Calculate total tasks dropped by SLA violations
        # (Total arrivals - successful starts = tasks starved in queue)
        num_drops = int(arrivals - exec_starts)

        # 3. Apply death penalty: force latency of all dropped tasks to timeout_ms
        if num_drops > 0:
            true_lats.extend([timeout_ms] * num_drops)

        # 4. Recompute P99 across the complete history of successful and dropped tasks
        if not true_lats:
            p99_start = timeout_ms
        else:
            p99_start = calc_p(true_lats, 0.99)

    return {
        "Experiment_Tag": exp_name,
        "scheduler": scheduler_name,
        "nodes": nodes,
        "rho": rho,
        "loss": loss,
        "hotspot_ratio": hotspot_ratio,
        "ray_hotspot_scale": ray_hotspot_scale,
        "flux_root_scale": flux_root_scale,
        "p99_arrival_to_start_ms": p99_start,
        "execution_starts_per_sec": 0 if is_crashed else (m.total_execution_starts / (max_time_ms / 1000.0)),
        "start_success_ratio": 0 if is_crashed else (m.total_execution_starts / arrivals),
        "p99_decision_us_measured": p99_dec,
        "control_work_per_success_us": cw_per_success,
        "control_actions_per_task": m.total_probe_actions / arrivals,
        "false_optimism_rate": m.total_false_optimism_events / max(1, m.total_probe_actions),
        "reservation_hold_time_ms": (
            sum(m.reservation_hold_times) / max(1, len(m.reservation_hold_times))
        ),
        "pending_expiry_ratio": m.total_pending_expiry / reservations,
        "duplicate_da_count": m.total_duplicate_das,
        "late_winner_residence_ms": (
            sum(m.late_winner_residence_times) / max(1, len(m.late_winner_residence_times))
        ),
        "uniqueness_violation_count": m.total_uniqueness_violations,
        "honest_start_goodput": m.total_honest_goodput,

        "local_service_p50_us": local_stats["p50"],
        "local_service_p95_us": local_stats["p95"],
        "local_service_p99_us": local_stats["p99"],
        "local_service_mean_us": local_stats["mean"],
        "local_service_count": local_stats["count"],

        "gcs_service_p50_us": gcs_service_stats["p50"],
        "gcs_service_p95_us": gcs_service_stats["p95"],
        "gcs_service_p99_us": gcs_service_stats["p99"],
        "gcs_service_mean_us": gcs_service_stats["mean"],
        "gcs_service_count": gcs_service_stats["count"],

        "gcs_wait_p50_us": gcs_wait_stats["p50"],
        "gcs_wait_p95_us": gcs_wait_stats["p95"],
        "gcs_wait_p99_us": gcs_wait_stats["p99"],
        "gcs_wait_mean_us": gcs_wait_stats["mean"],
        "gcs_wait_count": gcs_wait_stats["count"],

        # Ensure plotting scripts can access X-axis, slicing variables, and time-series arrays safely
        "squatter_ratio": float(getattr(wc, "squatter_ratio", 0.0)),
        "chaos_kill_ratio": float(params.get("chaos_kill_ratio", 0.0)),
        "da_drift_sigma": float(getattr(lam_cfg, "da_drift_sigma", 0.8)) if lam_cfg else 0.8,
        "teg_gamma": float(getattr(lam_cfg, "teg_gamma", 1.0)) if lam_cfg else 1.0,
        "enable_missingness_guard": bool(getattr(lam_cfg, "enable_missingness_guard", True)) if lam_cfg else True,
        "enable_two_phase": bool(getattr(lam_cfg, "enable_two_phase", True)) if lam_cfg else False,
        "enable_regeneration": bool(getattr(wc, "enable_regeneration", True)),
        "timeseries": timeseries_log,

        "ray_local_attempts": getattr(scheduler, "stat_local_attempts", 0),
        "ray_local_hits": getattr(scheduler, "stat_local_hits", 0),
        "ray_local_to_gcs": getattr(scheduler, "stat_local_to_gcs", 0),
        "ray_local_commit_failures": getattr(scheduler, "stat_local_commit_failures", 0),
        "ray_gcs_enqueues": getattr(scheduler, "stat_gcs_enqueues", 0),
        "ray_gcs_process": getattr(scheduler, "stat_gcs_process", 0),
        "ray_gcs_hits": getattr(scheduler, "stat_gcs_hits", 0),
        "ray_gcs_commit_failures": getattr(scheduler, "stat_gcs_commit_failures", 0),
        "ray_spillbacks": getattr(scheduler, "stat_spillbacks", 0),
        "ray_drops": getattr(scheduler, "stat_drops", 0),

        "local_hit_ratio": getattr(scheduler, "stat_local_hits", 0) / max(1, getattr(scheduler, "stat_local_attempts", 0)),
        "local_to_gcs_ratio": getattr(scheduler, "stat_local_to_gcs", 0) / max(1, getattr(scheduler, "stat_local_attempts", 0)),
        "gcs_process_ratio": getattr(scheduler, "stat_gcs_process", 0) / max(1, getattr(scheduler, "stat_gcs_enqueues", 0)),
        "gcs_hit_ratio": getattr(scheduler, "stat_gcs_hits", 0) / max(1, getattr(scheduler, "stat_gcs_enqueues", 0)),
        "gcs_commit_fail_ratio": getattr(scheduler, "stat_gcs_commit_failures", 0) / max(1, getattr(scheduler, "stat_gcs_enqueues", 0)),
        "spillback_ratio": getattr(scheduler, "stat_spillbacks", 0) / max(1, getattr(scheduler, "stat_gcs_enqueues", 0)),
        "gcs_enqueues_per_start": getattr(scheduler, "stat_gcs_enqueues", 0) / max(1, m.total_execution_starts),
        "drops_per_start": getattr(scheduler, "stat_drops", 0) / max(1, m.total_execution_starts),
    }