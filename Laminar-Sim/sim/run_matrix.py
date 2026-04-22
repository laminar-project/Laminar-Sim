import sys
import os
import csv
import concurrent.futures
import multiprocessing


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.run_exp import run_single



def execute_task(args):
    try:
        return True, args[0], run_single(args[0], args[1]), None
    except Exception as e:
        return False, args[0], None, str(e)


def load_done_tags(csv_path):
    done = set()

    if not os.path.exists(csv_path):
        return done

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row.get("Experiment_Tag") or row.get("ExperimentTag")
            if tag:
                done.add(tag)

    return done



def sweep():
    tasks = []
    # Unified realism knobs for all experiments
    baseline_realism = {
        "loss": 0.0,
        "hotspot_ratio": 0.2,
    }

    # Exp1: Mixed load
    for rho in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for s in ["laminar", "slurm", "ray", "flux"]:
            params = {
                "scheduler": s,
                "nodes": 5000,
                "rho": rho,
                "max_time_ms": 5000.0,
                **baseline_realism,
            }

            if s == "laminar":
                params.update({
                    "enable_taylor": True,
                    "enable_missingness_guard": True,
                    "enable_two_phase": False,
                    "enable_regeneration": True,
                    "hetero_zones": True,
                    "target_zone_size": 256,
                    "zone_jitter": 0.20,
                })

            tasks.append((
                f"Exp1_MixedLoad_rho{rho}_{s}",
                params
            ))


    # =================================================================
    # Exp2: Scale-out O(1)
    # Validation: Demonstrate Laminar's near-O(1) control overhead as scale increases
    # =================================================================
    for nodes in [5000, 10000, 20000, 32000]:
        tasks.append((
            f"Exp2_ScaleO1_nodes{nodes}_laminar",
            {
                "scheduler": "laminar",
                "nodes": nodes,
                "rho": 0.8,
                "max_time_ms": 5000.0,
                **baseline_realism,
                "hetero_zones": True,
                "target_zone_size": 256,
                "zone_jitter": 0.20,
                "enable_taylor": True,
                "enable_missingness_guard": True,
                "enable_two_phase": False,
                "enable_regeneration": True,
            }
        ))

    # ================================================================
    # Exp 3: State Staleness and Partial Visibility Sweep
    # Objective: Sweep global state synchronization delay. Observe whether stale views 
    # trigger thundering herds and retry storms across different control planes.
    # ================================================================
    for delay_ms in [0.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        for sched in ["laminar"]:
            tasks.append((
                f"Exp3_Staleness_{sched}_delay{delay_ms}",
                {
                    "scheduler": sched,
                    "nodes": 5000,
                    # Must maintain sufficiently high load and hotspots; otherwise, stale views will not induce herding.
                    "rho": 0.8,
                    "hotspot_ratio": 0.35,  
                    
                    # Core perturbation variable: physical hooks injected into the underlying layers
                    "baseline_heartbeat_ms": delay_ms,  # Synchronization period for Ray GCS / Flux Root
                    "laminar_zhaf_sync_ms": delay_ms,   # Synchronization period for Laminar Z-HAF
                    
                    "max_time_ms": 5000.0,
                }
            ))



    # =================================================================
    # Exp 4a: Malicious Squatter Containment (Pending Containment & Two Phase)
    # Sweep malicious squatter ratio: 0% to 10%
    # =================================================================
    for sq in [0.0, 0.05, 0.1]:
        for tp in [True, False]:
            tasks.append((
                f"Exp4b_Pending_sq{sq}_TP{tp}",
                {
                    "scheduler": "laminar",
                    "nodes": 5000,
                    "rho": 0.5,
                    "loss": 0.0,
                    "squatter_ratio": sq,              # Squatter ratio as independent variable
                    "enable_taylor": True,
                    "enable_missingness_guard": True,
                    "enable_two_phase": tp,            # TTL termination toggle
                    "enable_regeneration": False,      # Isolate variable: disable regeneration
                    "max_time_ms": 5000.0,
                }
            ))

    # =================================================================
    # Exp 4b: UDP Probe Loss Limits (Packet Loss & Regeneration)
    # Sweep network loss rate: 0% to 30%
    # =================================================================
    for loss in [0.0, 0.1, 0.2, 0.3]:
        for regen in [True, False]:
            tasks.append((
                f"Exp4c_Regen_loss{loss}_Regen{regen}",
                {
                    "scheduler": "laminar",
                    "nodes": 5000,
                    "rho": 0.8,
                    "loss": loss,                      # Network packet loss rate as independent variable
                    "squatter_ratio": 0.0,
                    "enable_taylor": True,
                    "enable_missingness_guard": True,
                    "enable_two_phase": False,         # Isolate variable: disable two-phase reservation
                    "enable_regeneration": regen,      # Timeout regeneration fallback toggle
                    "max_time_ms": 5000.0,
                }
            ))

    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "master_evaluation_stream.csv")

    done_tags = load_done_tags(csv_path)
    tasks = [task for task in tasks if task[0] not in done_tags]

    print(f"[INFO] already done: {len(done_tags)}")
    print(f"[INFO] remaining: {len(tasks)}")

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, multiprocessing.cpu_count() - 5)
    ) as exe, open(csv_path, "a", newline="") as f:


        writer = None


        futures = {
            exe.submit(execute_task, task): task[0]
            for task in tasks
        }


        for fut in concurrent.futures.as_completed(futures):
            try:
                ok, name, res, err = fut.result()
            except Exception as e:
                name = futures[fut]
                print(f"[FAIL] {name}: {e}")
                continue


            if ok:
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=sorted(res.keys()))
                    if not file_exists:
                        writer.writeheader()
                        file_exists = True
                        f.flush()
                        os.fsync(f.fileno())


                writer.writerow(res)
                f.flush()
                os.fsync(f.fileno())
                print(f"[OK] {name} ok")
            else:
                print(f"[FAIL] {name}: {err}")



if __name__ == "__main__":
    multiprocessing.freeze_support()
    sweep()