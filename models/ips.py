"""
ips.py
──────
Policy Enforcement module:

    "Policy Enforcement (IPS & Protector): Converts the intelligence derived
     from the ML engine into active defense rules, such as dynamic blocklisting
     or ARP spoofing, to isolate attackers from the production network."

This module implements the closed-loop feedback component that sits between
the ML Analysis Engine and the production network.  It consumes anomaly
decisions from KCenterClustering.predict() and translates them into:

  1. Dynamic IP blocklist  — source IPs flagged as anomalous are added to a
                             deny-list that the IDS/firewall enforces.  Entries
                             expire after TTL seconds to avoid stale blocks.

  2. ARP spoofing trigger  — for attackers already inside the network segment,
                             the Protector issues ARP replies that redirect the
                             attacker's traffic to a honeypot rather than the
                             real production asset (identity verification via
                             ARP, as described in Ahn et al. [2019]).

  3. Rotation notification — when CamouflageNet rotates honeypot IPs, the IPS
                             is notified so it can invalidate cached attacker
                             scan results stored in the blocklist.

Paper result (Table 5 — ablation):
  Full system (IPS active)  : TTI = 404 s
  IPS disabled              : TTI = 215 s
  → IPS contributes ~189 s of additional attacker delay by blocking resumed scans.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


# ─── Block entry ──────────────────────────────────────────────────────────────

@dataclass
class BlockEntry:
    """A single entry in the dynamic blocklist."""
    src_ip:        str
    blocked_at:    float          # Unix timestamp
    reason:        str            # e.g. "k-center anomaly", "ARP spoof trigger"
    cluster_dist:  float          # distance-to-nearest-centre at detection time
    ttl:           float          # block lifetime in seconds (0 = permanent)
    arp_spoofed:   bool = False   # True if ARP redirect has been issued

    @property
    def is_expired(self) -> bool:
        if self.ttl <= 0:
            return False
        return (time.time() - self.blocked_at) > self.ttl


# ─── IPS ──────────────────────────────────────────────────────────────────────

class IPSProtector:
    """
    Intrusion Prevention System — the Policy Enforcement layer described.
    The IPS receives anomaly decisions from the ML Analysis Engine and issues
    active defense rules to isolate the detected attacker.

    Usage
    -----
    >>> ips = IPSProtector()
    >>> # After k-center flags a session as anomalous:
    >>> blocked = ips.process_ml_decision(
    ...     src_ip="192.168.1.55", is_anomaly=True,
    ...     cluster_dist=0.43, honeypot_ip="10.1.0.112"
    ... )
    >>> print(ips.is_blocked("192.168.1.55"))  # True

    Rotation integration
    --------------------
    >>> ips.on_rotation_event(rotated_honeypot_ips=["10.1.0.112", "10.1.0.113"])
    """

    def __init__(
        self,
        block_ttl:           float = 3600.0,  # 1-hour default block lifetime
        arp_spoof_threshold: float = 0.0,     # block any flagged IP (no extra threshold)
        verbose:             bool  = False,
    ):
        """
        Parameters
        ----------
        block_ttl           : how long (seconds) a blocked IP stays on the
                              deny-list before being eligible for re-evaluation
                              (0 = permanent block)
        arp_spoof_threshold : minimum anomaly distance to also trigger ARP
                              spoofing (default 0 = spoof all flagged IPs)
        verbose             : print block/unblock events
        """
        self.block_ttl           = block_ttl
        self.arp_spoof_threshold = arp_spoof_threshold
        self.verbose             = verbose

        self._blocklist: Dict[str, BlockEntry] = {}
        self._arp_redirects: Dict[str, str]    = {}  # src_ip → honeypot_ip
        self._rotation_events: List[float]     = []  # timestamps of rotations
        self._total_blocked:   int             = 0
        self._total_arp_spoof: int             = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Core API
    # ──────────────────────────────────────────────────────────────────────────

    def process_ml_decision(
        self,
        src_ip:       str,
        is_anomaly:   bool,
        cluster_dist: float,
        honeypot_ip:  Optional[str] = None,
    ) -> bool:
        """
        Process an anomaly decision from the ML Analysis Engine.

        If is_anomaly is True:
          • Add src_ip to the dynamic blocklist
          • If cluster_dist >= arp_spoof_threshold AND a honeypot_ip is provided,
            issue an ARP redirect (simulated)

        Returns
        -------
        blocked : True if the IP was added to (or was already on) the blocklist
        """
        if not is_anomaly:
            return False

        self._add_to_blocklist(src_ip, cluster_dist)

        if honeypot_ip and cluster_dist >= self.arp_spoof_threshold:
            self._issue_arp_spoof(src_ip, honeypot_ip)

        return True

    def process_batch(
        self,
        src_ips:       np.ndarray,    # (n,) string array of source IPs
        y_pred:        np.ndarray,    # (n,) binary predictions from k-center
        cluster_dists: np.ndarray,    # (n,) distance-to-nearest-centre scores
        honeypot_ips:  Optional[List[str]] = None,
    ) -> int:
        """
        Batch version of process_ml_decision.  Processes an entire session
        batch from the ML engine.

        Returns
        -------
        n_blocked : number of new IPs added to the blocklist this batch
        """
        n_blocked = 0
        for i, (ip, pred, dist) in enumerate(zip(src_ips, y_pred, cluster_dists)):
            hp = honeypot_ips[i] if honeypot_ips else None
            if self.process_ml_decision(ip, bool(pred == 1), float(dist), hp):
                n_blocked += 1
        return n_blocked

    def is_blocked(self, src_ip: str) -> bool:
        """Return True if src_ip is currently on the active blocklist."""
        self._expire_stale_entries()
        return src_ip in self._blocklist

    def get_block_entry(self, src_ip: str) -> Optional[BlockEntry]:
        """Return the BlockEntry for src_ip, or None if not blocked."""
        self._expire_stale_entries()
        return self._blocklist.get(src_ip)

    def get_arp_redirect(self, src_ip: str) -> Optional[str]:
        """
        Return the honeypot IP to which src_ip's traffic is redirected,
        or None if no ARP spoof is active for this IP.
        """
        return self._arp_redirects.get(src_ip)

    # ──────────────────────────────────────────────────────────────────────────
    # Rotation integration
    # ──────────────────────────────────────────────────────────────────────────

    def on_rotation_event(self, rotated_honeypot_ips: List[str]) -> int:
        """
        Called by CamouflageNet when honeypots rotate their IPs.

        Per the paper's ablation study: when the IPS is active, previously
        identified attackers that resume scanning after a rotation are
        immediately re-blocked because the IPS retains their src_ip in the
        blocklist.  This is what keeps TTI at 404 s (full system) vs 215 s
        (IPS disabled).

        This method:
          • Logs the rotation event
          • Invalidates any ARP redirects pointing at rotated IPs (the
            honeypot IP has changed, so the old redirect is stale)
          • Returns the number of ARP redirects invalidated

        Parameters
        ----------
        rotated_honeypot_ips : list of old honeypot IPs that have now changed

        Returns
        -------
        n_invalidated : number of ARP redirect entries cleared
        """
        self._rotation_events.append(time.time())
        rotated_set = set(rotated_honeypot_ips)
        to_clear = [
            ip for ip, hp in self._arp_redirects.items()
            if hp in rotated_set
        ]
        for ip in to_clear:
            del self._arp_redirects[ip]
        if self.verbose and to_clear:
            print(f"  [IPS] Rotation: cleared {len(to_clear)} stale ARP redirects")
        return len(to_clear)

    # ──────────────────────────────────────────────────────────────────────────
    # Blocklist management
    # ──────────────────────────────────────────────────────────────────────────

    def unblock(self, src_ip: str) -> bool:
        """Manually remove an IP from the blocklist (e.g. after investigation)."""
        removed = self._blocklist.pop(src_ip, None)
        self._arp_redirects.pop(src_ip, None)
        if removed and self.verbose:
            print(f"  [IPS] Unblocked {src_ip}")
        return removed is not None

    def get_active_blocklist(self) -> List[BlockEntry]:
        """Return all currently active (non-expired) block entries."""
        self._expire_stale_entries()
        return list(self._blocklist.values())

    def summary(self) -> Dict:
        """Return a summary of IPS state for reporting."""
        self._expire_stale_entries()
        return {
            "active_blocks":     len(self._blocklist),
            "active_arp_spoof":  len(self._arp_redirects),
            "total_blocked":     self._total_blocked,
            "total_arp_spoof":   self._total_arp_spoof,
            "rotation_events":   len(self._rotation_events),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _add_to_blocklist(self, src_ip: str, cluster_dist: float) -> None:
        if src_ip not in self._blocklist:
            entry = BlockEntry(
                src_ip=src_ip,
                blocked_at=time.time(),
                reason="k-center anomaly",
                cluster_dist=cluster_dist,
                ttl=self.block_ttl,
            )
            self._blocklist[src_ip] = entry
            self._total_blocked += 1
            if self.verbose:
                print(f"  [IPS] Blocked {src_ip}  (dist={cluster_dist:.4f})")

    def _issue_arp_spoof(self, src_ip: str, honeypot_ip: str) -> None:
        """
        Simulate issuing an ARP reply that redirects src_ip's traffic to
        honeypot_ip rather than the real production asset.

        In the real Mininet testbed this would call:
            subprocess.run(["arpspoof", "-i", iface, "-t", src_ip, honeypot_ip])
        Here we record the redirect in a dict for simulation purposes.
        """
        self._arp_redirects[src_ip] = honeypot_ip
        if src_ip in self._blocklist:
            self._blocklist[src_ip].arp_spoofed = True
        self._total_arp_spoof += 1
        if self.verbose:
            print(f"  [IPS] ARP spoof: {src_ip} → {honeypot_ip}")

    def _expire_stale_entries(self) -> None:
        """Remove blocklist entries whose TTL has elapsed."""
        expired = [ip for ip, e in self._blocklist.items() if e.is_expired]
        for ip in expired:
            del self._blocklist[ip]
            self._arp_redirects.pop(ip, None)
            if self.verbose:
                print(f"  [IPS] Block expired: {ip}")


# ─── Quick demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ips = IPSProtector(block_ttl=3600, verbose=True)

    # Simulate ML engine flagging two attackers
    ips.process_ml_decision("192.168.1.55", is_anomaly=True,
                             cluster_dist=0.43, honeypot_ip="10.1.0.112")
    ips.process_ml_decision("192.168.1.77", is_anomaly=True,
                             cluster_dist=0.61, honeypot_ip="10.1.0.115")
    ips.process_ml_decision("192.168.1.10", is_anomaly=False,
                             cluster_dist=0.05)

    print(f"\nBlocked 192.168.1.55: {ips.is_blocked('192.168.1.55')}")
    print(f"Blocked 192.168.1.10: {ips.is_blocked('192.168.1.10')}")
    print(f"ARP redirect for .55: {ips.get_arp_redirect('192.168.1.55')}")

    # Simulate a CamouflageNet rotation
    n_cleared = ips.on_rotation_event(["10.1.0.112", "10.1.0.113"])
    print(f"\nARP redirects cleared after rotation: {n_cleared}")
    print(f"ARP redirect for .55 after rotation : {ips.get_arp_redirect('192.168.1.55')}")

    print("\nIPS Summary:", ips.summary())
