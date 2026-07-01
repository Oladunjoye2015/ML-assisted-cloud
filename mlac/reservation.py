"""Fill-aware trade-cap bookkeeping (AUDIT FIX H2), as a self-contained, testable unit.

This is the reference implementation of the logic embedded in the live service's
reserve_trade_slot()/confirm_fill(): a /predict recommendation takes a PENDING slot;
a confirmed fill retires it and becomes authoritative; unfilled reservations expire
after a TTL so a recommendation that never becomes an order can't consume the cap.

Caps are evaluated on the EFFECTIVE count (confirmed fills + active reservations).
Thread-safety in the live service is provided by its _STATE_LOCK; this class is the
pure logic and takes an injectable clock for deterministic testing.
"""
from __future__ import annotations
from collections import deque
from typing import Callable, Deque, Dict, Optional, Tuple
import time as _time


class ReservationBook:
    def __init__(
        self,
        max_open: int,
        max_per_day_total: int,
        max_per_day_pair: int,
        ttl_seconds: float = 300.0,
        clock: Callable[[], float] = _time.time,
    ):
        self.max_open = max_open
        self.max_total = max_per_day_total
        self.max_pair = max_per_day_pair
        self.ttl = ttl_seconds
        self._clock = clock
        self._confirmed: Dict[str, int] = {}
        self._pending: Dict[str, Deque[Tuple[float, Optional[str]]]] = {}
        self._open_ids: set[str] = set()

    # --- internal ---------------------------------------------------------
    def _prune(self) -> None:
        now = self._clock()
        for pair, q in list(self._pending.items()):
            while q and (now - q[0][0] > self.ttl):
                q.popleft()
            if not q:
                self._pending.pop(pair, None)

    def _pending_count(self, pair: Optional[str] = None) -> int:
        if pair is not None:
            return len(self._pending.get(pair, ()))
        return sum(len(q) for q in self._pending.values())

    # --- public -----------------------------------------------------------
    def effective_pair(self, pair: str) -> int:
        self._prune()
        return self._confirmed.get(pair, 0) + self._pending_count(pair)

    def effective_total(self) -> int:
        self._prune()
        return sum(self._confirmed.values()) + self._pending_count()

    def effective_open(self) -> int:
        self._prune()
        return len(self._open_ids) + self._pending_count()

    def reserve(self, pair: str, fingerprint: Optional[str] = None) -> Optional[str]:
        """Return None if a slot was reserved, else a reason string."""
        self._prune()
        if sum(self._confirmed.values()) + self._pending_count() >= self.max_total:
            return "daily_total_cap"
        if self._confirmed.get(pair, 0) + self._pending_count(pair) >= self.max_pair:
            return "daily_pair_cap"
        if len(self._open_ids) + self._pending_count() >= self.max_open:
            return "open_trade_cap"
        self._pending.setdefault(pair, deque()).append((self._clock(), fingerprint))
        return None

    def confirm_fill(self, pair: str, tracking_key: Optional[str] = None,
                     fingerprint: Optional[str] = None) -> None:
        """A real fill is authoritative: count it and retire one matching reservation."""
        self._prune()
        self._confirmed[pair] = self._confirmed.get(pair, 0) + 1
        if tracking_key is not None:
            self._open_ids.add(str(tracking_key))
        q = self._pending.get(pair)
        if q:
            if fingerprint is not None:
                for i, (_, fp) in enumerate(q):
                    if fp == fingerprint:
                        del q[i]
                        break
                else:
                    q.popleft()
            else:
                q.popleft()
            if not q:
                self._pending.pop(pair, None)

    def close_fill(self, tracking_key: str) -> None:
        self._open_ids.discard(str(tracking_key))

    def reset_day(self) -> None:
        self._confirmed.clear()
        self._pending.clear()
