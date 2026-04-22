# sim/baselines/config.py

from dataclasses import dataclass

@dataclass
class BaselineConfig:
    """
    Configuration for centralized and hierarchical baselines.

    All parameters are deliberately optimistic to provide Slurm / Ray / Flux 
    with a highly favorable upper bound. The goal is to compare theoretical 
    control-plane limits against Laminar rather than to artificially handicap 
    the baselines.
    """

    # Unified failure threshold: If any scheduling decision exceeds this duration, the control plane is considered crashed
    timeout_crash_us: float = 5000_000.0  

    # Toggle crash-on-timeout model and scalar filtering
    enable_crash: bool = True
    enable_scalar_filter: bool = False

    # ------------------------------------------------------------------
    # 1. Slurm / K8s-style centralized global queue
    # ------------------------------------------------------------------
    slurm_base_scan_us: float = 0.1
    slurm_per_node_us: float = 0.01
    slurm_lock_base_us: float = 0.5
    slurm_lock_scale: float = 10000.0
    slurm_max_retries: int = 3
    slurm_retry_base_ms: float = 2.0
    slurm_max_queue: int = 200000

    # ------------------------------------------------------------------
    # 2. Ray-style hierarchical (local-first, GCS spillback)
    # ------------------------------------------------------------------
    ray_local_k: int = 8                
    ray_local_base_us: float = 20.0        
    ray_gcs_base_us: float = 50.0         
    ray_gcs_shards: int = 32              
    ray_gcs_hotspot_scale: float = 500.0  
    ray_gcs_shard_bias: float = 0.5       
    ray_max_spillbacks: int = 16          
    ray_retry_base_ms: float = 10.0       
    ray_network_rtt_us: float = 500.0     
    ray_gcs_max_queue: int = 200000

    # ------------------------------------------------------------------
    # 3. Flux-style hierarchical broker tree
    # ------------------------------------------------------------------
    flux_tree_fanout: int = 16           
    flux_match_base_us: float = 2.0      
    flux_per_node_us: float = 0.005       
    flux_hop_delay_us: float = 500.0       
    flux_root_scale: float = 4000.0      
    flux_max_retries: int = 5            
    flux_retry_base_ms: float = 10.0      
    flux_leaf_k: int = 32
    flux_root_max_queue: int = 200000

    # Global physical constraint parameters
    baseline_heartbeat_ms: float = 10.0  # Stale-view heartbeat sync interval