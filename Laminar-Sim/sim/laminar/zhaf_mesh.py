import random

class ZHAFMesh:
    def __init__(self, env, state, config) -> None:
        self.env, self.state, self.cfg = env, state, config
        self.view_node_S, self.view_node_H, self.view_node_A = list(state.node_s), list(state.node_h), list(state.node_masks)
        self.view_node_S_dot, self.view_node_H_dot, self.view_node_ts = [0.0]*state.num_nodes, [0.0]*state.num_nodes, [0.0]*state.num_nodes
        self.view_zone_S, self.view_zone_H, self.view_zone_ts = list(state.zone_s), list(state.zone_h), [0.0]*state.num_zones
        self._last_S, self._last_H, self._last_time = list(state.node_s), list(state.node_h), 0.0
        self.dead_nodes = set() # Target list for receiving external network disconnection signals
        self.env.schedule(self.cfg.zhaf_broadcast_ms, self._tick)

    def _tick(self) -> None:
        now, dt = self.env.now, max(1e-9, self.env.now - self._last_time)
        c_S, c_H, c_A = list(self.state.node_s), list(self.state.node_h), list(self.state.node_masks)
        c_zS, c_zH = list(self.state.zone_s), list(self.state.zone_h)
        d_S = [(c_S[i] - self._last_S[i]) / dt for i in range(self.state.num_nodes)]
        d_H = [(c_H[i] - self._last_H[i]) / dt for i in range(self.state.num_nodes)]
        self._last_S, self._last_H, self._last_time = c_S, c_H, now
        
        loss = float(getattr(self.cfg, "packet_loss_rate", 0.0))
        
        # Chaos injection defense: fetch dead node list to prevent null pointers
        dead_set = getattr(self, "dead_nodes", set())
        
        # Physical NIC constraint: only nodes without packet loss and not in the dead list can broadcast status
        m_nodes = [
            (random.random() >= loss) and (i not in dead_set) 
            for i in range(self.state.num_nodes)
        ]
        m_zones = [random.random() >= loss for _ in range(self.state.num_zones)]
        
        self.env.schedule(self.cfg.network_rtt_ms, self._apply_bulk_view, c_S, c_H, c_A, d_S, d_H, m_nodes, c_zS, c_zH, m_zones, now)
        self.env.schedule(self.cfg.zhaf_broadcast_ms, self._tick)

    def _apply_bulk_view(self, c_S, c_H, c_A, d_S, d_H, m_nodes, c_zS, c_zH, m_zones, ts: float) -> None:
        for i in range(self.state.num_nodes):
            if m_nodes[i]:
                self.view_node_S[i], self.view_node_H[i], self.view_node_A[i] = c_S[i], c_H[i], c_A[i]
                self.view_node_S_dot[i], self.view_node_H_dot[i], self.view_node_ts[i] = d_S[i], d_H[i], ts
        for i in range(self.state.num_zones):
            if m_zones[i]:
                self.view_zone_S[i], self.view_zone_H[i], self.view_zone_ts[i] = c_zS[i], c_zH[i], ts

    def get_zone_aggregates(self) -> tuple:
        z_s_out, z_h_out, now = [], [], self.env.now
        guard = getattr(self.cfg, "enable_missingness_guard", True)
        # Adjust threshold to 1.5x to ensure degradation protection triggers under realistic packet loss
        limit_ms = float(getattr(self.cfg, "zhaf_broadcast_ms", 50.0)) * 1.5 
        for i in range(self.state.num_zones):
            tau, s, h = now - self.view_zone_ts[i], self.view_zone_S[i], self.view_zone_H[i]
            if guard and tau > limit_ms: s, h = 0.0, h + 1000.0 
            z_s_out.append(float(s)); z_h_out.append(float(h))
        return z_s_out, z_h_out

    def get_projected_node_view(self, indices: list) -> tuple:
        s_out, h_out, a_out, now = [], [], [], self.env.now
        taylor, guard = getattr(self.cfg, "enable_taylor", True), getattr(self.cfg, "enable_missingness_guard", True)
        limit_ms = float(getattr(self.cfg, "zhaf_broadcast_ms", 50.0)) * 1.5
        for i in indices:
            tau, s, h, a = now - self.view_node_ts[i], self.view_node_S[i], self.view_node_H[i], self.view_node_A[i]
            if guard and tau > limit_ms: 
                s, h, a = 0.0, h + 100.0, 0
            elif taylor: 
                # Strictly limit Taylor expansion ceiling! Predictions must not exceed the physical maximum of 256.0
                s = min(256.0, max(0.0, s + tau * self.view_node_S_dot[i]))
                h = max(0.0, h + tau * self.view_node_H_dot[i])
            s_out.append(float(s)); h_out.append(float(h)); a_out.append(int(a))
        return s_out, h_out, a_out