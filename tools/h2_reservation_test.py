import threading, time
from collections import deque

MAX_PAIR=1; MAX_TOTAL=3; MAX_OPEN=1; TTL=1  # 1s TTL for the test
lock=threading.RLock()
confirmed={}; pending={}; open_ids=set()

def now(): return time.time()
def prune():
    t=now()
    for p,q in list(pending.items()):
        while q and t-q[0][0]>TTL: q.popleft()
        if not q: pending.pop(p,None)
def pcount(p=None):
    return len(pending.get(p,())) if p else sum(len(q) for q in pending.values())
def reserve(p,fp):
    with lock:
        prune()
        if confirmed.get(p,0)+pcount(p)>=MAX_PAIR: return "pair_cap"
        if sum(confirmed.values())+pcount()>=MAX_TOTAL: return "total_cap"
        if len(open_ids)+pcount()>=MAX_OPEN: return "open_cap"
        pending.setdefault(p,deque()).append((now(),fp)); return None
def confirm(p,fp=None):
    with lock:
        prune()
        confirmed[p]=confirmed.get(p,0)+1
        q=pending.get(p)
        if q: q.popleft()
        if q is not None and not q: pending.pop(p,None)

print("T1 reserve holds cap before fill:")
print("  reserve#1:", reserve("EURUSD","a"), "| reserve#2 (same pair):", reserve("EURUSD","b"), "(expect pair_cap)")

print("T2 fill converts pending->confirmed (no double count):")
confirm("EURUSD","a"); open_ids.add("t1")
with lock: print(f"  confirmed={confirmed.get('EURUSD')} pending={pcount('EURUSD')} (expect 1 and 0)")

print("T3 reset + TTL expiry frees a never-filled reservation:")
confirmed.clear(); pending.clear(); open_ids.clear()
print("  reserve:", reserve("GBPUSD","x"), "| immediate re-reserve:", reserve("GBPUSD","y"),"(expect pair_cap)")
time.sleep(TTL+0.3)
print("  after TTL, re-reserve:", reserve("GBPUSD","z"), "(expect None = slot freed)")

print("T4 open cap counts pending (can't oversubscribe open slots):")
confirmed.clear(); pending.clear(); open_ids.clear()
print("  reserve A:", reserve("EURUSD","1"), "| reserve B diff pair:", reserve("USDJPY","2"), "(expect open_cap)")
