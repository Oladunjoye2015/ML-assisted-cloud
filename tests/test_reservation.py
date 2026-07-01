from mlac.reservation import ReservationBook


class FakeClock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, dt): self.t += dt


def make_book(clock, ttl=300):
    return ReservationBook(max_open=1, max_per_day_total=3, max_per_day_pair=1,
                           ttl_seconds=ttl, clock=clock)


def test_reservation_holds_cap_before_fill():
    b = make_book(FakeClock())
    assert b.reserve("EURUSD") is None
    assert b.reserve("EURUSD") == "daily_pair_cap"     # per-pair cap already reserved


def test_fill_converts_pending_without_double_count():
    b = make_book(FakeClock())
    b.reserve("EURUSD", fingerprint="a")
    b.confirm_fill("EURUSD", tracking_key="t1", fingerprint="a")
    assert b.effective_pair("EURUSD") == 1             # 1 confirmed, 0 pending
    assert b._pending_count("EURUSD") == 0


def test_ttl_expiry_frees_slot():
    clk = FakeClock()
    b = make_book(clk, ttl=300)
    assert b.reserve("GBPUSD") is None
    assert b.reserve("GBPUSD") == "daily_pair_cap"
    clk.advance(301)                                   # reservation ages out
    assert b.reserve("GBPUSD") is None                 # slot freed


def test_open_cap_counts_pending():
    b = make_book(FakeClock())
    assert b.reserve("EURUSD") is None
    assert b.reserve("USDJPY") == "open_trade_cap"     # max_open=1, one pending already


def test_total_cap():
    b = ReservationBook(max_open=10, max_per_day_total=2, max_per_day_pair=5,
                        ttl_seconds=300, clock=FakeClock())
    assert b.reserve("EURUSD") is None
    assert b.reserve("EURUSD") is None
    assert b.reserve("EURUSD") == "daily_total_cap"


def test_close_fill_frees_open_slot():
    b = make_book(FakeClock())
    b.reserve("EURUSD", fingerprint="a")
    b.confirm_fill("EURUSD", tracking_key="t1", fingerprint="a")
    assert b.effective_open() == 1
    b.close_fill("t1")
    assert b.effective_open() == 0
