# sim/laminar/teg_gateway.py

import random


class TEGGateway:
    """
    Thermo-Economic Gateway (TEG), macro flow-splitting front-end.

    - Uses zone-level aggregates (Slack S_z, Heat H_z) from ZHAF.
    - Unified utility ratio form: W_z = (1 + S_z) / (1 + H_z)^gamma.
    - At gamma=1, W_z = (1 + S_z) / (1 + H_z).
    - Each probe is independently assigned a zone and a launchpad node.
    """

    def __init__(self, env, state, cfg, zhaf) -> None:
        self.env = env
        self.state = state
        self.cfg = cfg
        self.zhaf = zhaf

        # DA probe handle compatibility
        self.da_probe = None
        self.daprobe = None

        # Default paper semantics: gamma = 1
        self.gamma = 1.0

        # Batch processing
        self.pending_batch = []
        self.batch_interval_ms = 1.0

        # Start flush loop
        self.env.schedule(0.0, self.flush_batch_loop)

    # --------- Compatibility interfaces ---------

    def _get_da_probe(self):
        return self.da_probe if self.da_probe is not None else self.daprobe

    def submit_job(self, job) -> None:
        self.pending_batch.append(job)

    def submitjob(self, job) -> None:
        # Compatibility for legacy method names
        self.submit_job(job)

    # --------- Retrieve ZHAF view (compatibility for legacy/new naming) ---------

    def _get_zone_aggregates(self):
        if hasattr(self.zhaf, "get_zone_aggregates"):
            return self.zhaf.get_zone_aggregates()
        if hasattr(self.zhaf, "getzoneaggregates"):
            return self.zhaf.getzoneaggregates()
        raise AttributeError(
            "ZHAFMesh has neither get_zone_aggregates nor getzoneaggregates"
        )

    # --------- Weight calculation: W = (1+S)/(1+H)^gamma ---------

    def _compute_zone_weights(self):
        zS, zH = self._get_zone_aggregates()
        weights = []
        totalw = 0.0

        for sval, hval in zip(zS, zH):
            s = float(sval)
            h = float(hval)

            denom = (1.0 + h) ** self.gamma
            if denom <= 0.0:
                w = 0.0
            else:
                w = (1.0 + s) / denom

            if w < 0.0:
                w = 0.0

            weights.append(w)
            totalw += w

        return zS, zH, weights, totalw

    # --------- Sample zone / launchpad ---------

    def _sample_zone(self, num_zones: int, weights, totalw: float) -> int:
        if num_zones <= 0:
            return 0

        if totalw <= 0.0:
            return random.randrange(num_zones)

        r = random.uniform(0.0, totalw)
        acc = 0.0
        zone_id = num_zones - 1

        for idx, w in enumerate(weights):
            acc += float(w)
            if r <= acc:
                zone_id = idx
                break

        return zone_id

    def _sample_launchpad(self, zone_id: int) -> int:
        start_node = self.state.zone_offsets[zone_id]
        end_node = self.state.zone_offsets[zone_id + 1]

        if end_node > start_node:
            return random.randrange(start_node, end_node)
        return random.randrange(self.state.num_nodes)

    # --------- Main loop ---------

    def flush_batch_loop(self) -> None:
        # Schedule next iteration
        self.env.schedule(self.batch_interval_ms, self.flush_batch_loop)

        if not self.pending_batch:
            return

        batch, self.pending_batch = self.pending_batch, []

        zS, zH, weights, totalw = self._compute_zone_weights()
        num_zones = max(1, len(zS))
        fma_cost = float(getattr(self.cfg, "hw_fma_us_per_zone", 0.02929))
        
        # Read global UDP packet loss rate
        loss_rate = float(getattr(self.cfg, "packet_loss_rate", 0.0))

        da = self._get_da_probe()

        for job in batch:
            # Record coarse-grained zone-level FMA cost per job
            job.control_work_us += fma_cost

            # Physical injection: UDP packet loss during TEG DA probe dispatch
            if random.random() < loss_rate:
                # Probe physical annihilation; silently wait for Regenerator fallback
                continue

            zone_id = self._sample_zone(num_zones, weights, totalw)
            launchpad_node = self._sample_launchpad(zone_id)

            if da is not None:
                if hasattr(da, "arrive_at_launchpad"):
                    da.arrive_at_launchpad(job, launchpad_node)
                else:
                    da.arriveatlaunchpad(job, launchpad_node)

    # Compatibility for legacy method name
    def flushbatchloop(self) -> None:
        self.flush_batch_loop()