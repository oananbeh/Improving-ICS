"""
camouflage_net.py
─────────────────
Simulates the CamouflageNet dynamic reconfiguration mechanism.

Key behaviours:
  • 20 honeypot nodes rotate their IP and MAC addresses every 120 seconds.
  • When rotation occurs, an attacker's previously completed scan of a
    honeypot is invalidated — the node appears as a brand-new unknown host.
  • This forces repeated reconnaissance, increasing the attacker's
    Time-to-Identify (TTI) by ~234 % (121 s → 404 s on average).
  • Each rotation event is logged and can trigger a feature-level
    "rotation signature" that feeds back to the IPS.

Network layout (Table 2 in paper):
  45 total nodes: 12 PLCs, 8 RTUs, 5 HMIs, 20 Honeypots
"""

from __future__ import annotations
import numpy as np
import ipaddress
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    N_NODES, N_PLCS, N_RTUS, N_HMIS, N_HONEYPOTS,
    ROTATION_INTERVAL, RANDOM_SEED,
)


# ─── Network node ─────────────────────────────────────────────────────────────

@dataclass
class NetworkNode:
    node_id:     int
    node_type:   str          # "plc", "rtu", "hmi", "honeypot"
    ip:          str
    mac:         str
    is_honeypot: bool

    def __repr__(self):
        return (f"NetworkNode({self.node_type.upper()} id={self.node_id} "
                f"ip={self.ip} mac={self.mac})")


@dataclass
class RotationEvent:
    """Records a single IP/MAC rotation performed by CamouflageNet."""
    timestamp:   float
    node_id:     int
    old_ip:      str
    new_ip:      str
    old_mac:     str
    new_mac:     str
    rotation_num: int


# ─── CamouflageNet ────────────────────────────────────────────────────────────

class CamouflageNet:
    """
    Models the CamouflageNet dynamic high-interaction honeynet.

    The network is built with 45 nodes:
      - 25 real production nodes (PLCs, RTUs, HMIs) — never rotated
      - 20 honeypot nodes         — rotated every ROTATION_INTERVAL seconds

    Usage
    -----
    >>> cn = CamouflageNet()
    >>> cn.build_network()
    >>> cn.run_for(duration=3600)  # simulate 1 hour
    >>> print(cn.rotation_log)
    """

    BASE_SUBNET = "10.0.0."   # /24 production subnet (192.168.1.x in practice)
    HON_SUBNET  = "10.1.0."   # honeypot subnet

    def __init__(self, seed: int = RANDOM_SEED):
        self._rng           = np.random.default_rng(seed)
        self.nodes:          Dict[int, NetworkNode] = {}
        self.rotation_log:   List[RotationEvent]    = []
        self._rotation_num   = 0
        self._sim_time       = 0.0      # simulated seconds elapsed
        self._network_built  = False
        self._available_ips: List[str] = []

    # ── Network construction ──────────────────────────────────────────────────

    def build_network(self) -> "CamouflageNet":
        """Create all 45 nodes with initial IP/MAC assignments."""
        node_id = 0

        # Real production nodes
        for _ in range(N_PLCS):
            self.nodes[node_id] = NetworkNode(
                node_id=node_id, node_type="plc",
                ip=f"{self.BASE_SUBNET}{10 + node_id}",
                mac=self._random_mac(), is_honeypot=False,
            )
            node_id += 1

        for _ in range(N_RTUS):
            self.nodes[node_id] = NetworkNode(
                node_id=node_id, node_type="rtu",
                ip=f"{self.BASE_SUBNET}{30 + node_id}",
                mac=self._random_mac(), is_honeypot=False,
            )
            node_id += 1

        for _ in range(N_HMIS):
            self.nodes[node_id] = NetworkNode(
                node_id=node_id, node_type="hmi",
                ip=f"{self.BASE_SUBNET}{50 + node_id}",
                mac=self._random_mac(), is_honeypot=False,
            )
            node_id += 1

        # Honeypot pool — initial IPs in the honeypot subnet
        self._available_ips = [
            f"{self.HON_SUBNET}{100 + i}" for i in range(200)
        ]
        self._rng.shuffle(self._available_ips)

        for i in range(N_HONEYPOTS):
            ip = self._available_ips[i]
            self.nodes[node_id] = NetworkNode(
                node_id=node_id, node_type="honeypot",
                ip=ip, mac=self._random_mac(), is_honeypot=True,
            )
            node_id += 1

        self._used_ip_pool = set(
            self._available_ips[:N_HONEYPOTS]
        )
        self._free_ip_pool = self._available_ips[N_HONEYPOTS:]

        self._network_built = True
        return self

    # ── Rotation ──────────────────────────────────────────────────────────────

    def rotate(self) -> List[RotationEvent]:
        """
        Perform one IP/MAC rotation cycle for all honeypot nodes.
        This is called every ROTATION_INTERVAL seconds.
        Returns a list of RotationEvent objects.
        """
        if not self._network_built:
            raise RuntimeError("Call build_network() first.")

        events = []
        honeypots = [n for n in self.nodes.values() if n.is_honeypot]

        for node in honeypots:
            old_ip  = node.ip
            old_mac = node.mac

            # Assign a new IP from the free pool
            if self._free_ip_pool:
                new_ip = self._free_ip_pool.pop(0)
                self._free_ip_pool.append(old_ip)   # old IP becomes free
            else:
                # Fallback: generate a fresh random IP
                new_ip = f"{self.HON_SUBNET}{self._rng.integers(100, 255)}"

            new_mac = self._random_mac()

            node.ip  = new_ip
            node.mac = new_mac

            self._rotation_num += 1
            evt = RotationEvent(
                timestamp=self._sim_time,
                node_id=node.node_id,
                old_ip=old_ip,
                new_ip=new_ip,
                old_mac=old_mac,
                new_mac=new_mac,
                rotation_num=self._rotation_num,
            )
            self.rotation_log.append(evt)
            events.append(evt)

        return events

    # ── Simulation runner ─────────────────────────────────────────────────────

    def run_for(self, duration: float) -> int:
        """
        Advance simulation by `duration` seconds, rotating at every
        ROTATION_INTERVAL boundary.

        Returns
        -------
        n_rotations : total number of rotation cycles performed
        """
        if not self._network_built:
            self.build_network()

        end_time    = self._sim_time + duration
        n_rotations = 0
        next_rotate = self._sim_time + ROTATION_INTERVAL

        while self._sim_time < end_time:
            step = min(next_rotate, end_time) - self._sim_time
            self._sim_time += step

            if self._sim_time >= next_rotate:
                self.rotate()
                n_rotations += 1
                next_rotate += ROTATION_INTERVAL

        return n_rotations

    # ── Attacker interaction: redirect ───────────────────────────────────────

    def redirect_suspicious_traffic(
        self,
        src_ip: str,
        is_flagged: bool,
    ) -> Tuple[bool, Optional[str]]:
        """
        Simulate IDS traffic steering.

        If traffic from src_ip is flagged by the IDS:
          • Redirect metadata to CamouflageNet (return honeypot IP)
          • Production network remains unaffected

        Returns
        -------
        (redirected, honeypot_ip)
        """
        if not is_flagged:
            return False, None
        # Pick a random honeypot to receive the redirected traffic
        honeypots = [n for n in self.nodes.values() if n.is_honeypot]
        chosen = self._rng.choice(honeypots)
        return True, chosen.ip

    # ── Network snapshot ──────────────────────────────────────────────────────

    def get_active_ips(self) -> List[str]:
        """Return current IP list as seen by an external scanner."""
        return [n.ip for n in self.nodes.values()]

    def get_honeypot_ips(self) -> List[str]:
        return [n.ip for n in self.nodes.values() if n.is_honeypot]

    def get_real_ips(self) -> List[str]:
        return [n.ip for n in self.nodes.values() if not n.is_honeypot]

    def summary(self) -> Dict:
        return {
            "total_nodes":       len(self.nodes),
            "real_nodes":        sum(1 for n in self.nodes.values() if not n.is_honeypot),
            "honeypot_nodes":    sum(1 for n in self.nodes.values() if n.is_honeypot),
            "sim_time_s":        self._sim_time,
            "total_rotations":   self._rotation_num,
            "rotation_events":   len(self.rotation_log),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _random_mac(self) -> str:
        octets = self._rng.integers(0, 256, size=6)
        # Ensure unicast and locally administered
        octets[0] = (octets[0] & 0xFE) | 0x02
        return ":".join(f"{o:02x}" for o in octets)


# ─── Quick demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cn = CamouflageNet()
    cn.build_network()
    print("Initial honeypot IPs:", cn.get_honeypot_ips()[:5], "...")
    n = cn.run_for(duration=3600)   # 1 hour
    print(f"\nAfter 3600 s: {n} rotation cycles, "
          f"{len(cn.rotation_log)} individual node rotations")
    print("Honeypot IPs after rotation:", cn.get_honeypot_ips()[:5], "...")
    print("\nSummary:", cn.summary())
