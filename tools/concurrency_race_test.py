import threading, time, random

MAX_TOTAL=3; MAX_PAIR=1; MAX_OPEN=1

def make_state():
    return {"count":{}, "open":set(), "lock":threading.RLock()}

# NEW: atomic reserve under lock (mirrors reserve_trade_slot)
def reserve_locked(s, pair):
    with s["lock"]:
        if sum(s["count"].values())>=MAX_TOTAL: return "total"
        if s["count"].get(pair,0)>=MAX_PAIR: return "pair"
        if len(s["open"])>=MAX_OPEN: return "open"
        time.sleep(random.uniform(0,0.001))  # widen the window
        s["count"][pair]=s["count"].get(pair,0)+1
        return None

# OLD: check-then-increment WITHOUT lock (the bug)
def reserve_unlocked(s, pair):
    if sum(s["count"].values())>=MAX_TOTAL: return "total"
    if s["count"].get(pair,0)>=MAX_PAIR: return "pair"
    if len(s["open"])>=MAX_OPEN: return "open"
    time.sleep(random.uniform(0,0.001))
    s["count"][pair]=s["count"].get(pair,0)+1
    return None

def run(fn, threads=50):
    s=make_state(); results=[]
    def worker(): results.append(fn(s,"EURUSD"))
    ts=[threading.Thread(target=worker) for _ in range(threads)]
    for t in ts: t.start()
    for t in ts: t.join()
    reserved=sum(1 for r in results if r is None)
    return reserved, s["count"].get("EURUSD",0)

for name,fn in [("UNLOCKED (old bug)",reserve_unlocked),("LOCKED (fix)",reserve_locked)]:
    res=[run(fn) for _ in range(20)]
    over=[r for r in res if r[0]>MAX_PAIR]
    worst=max(r[0] for r in res)
    print(f"{name:22} per-pair cap={MAX_PAIR} | worst reservations in a run={worst} | runs breaching cap={len(over)}/20")
