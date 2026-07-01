#!/usr/bin/env python3
"""Download fresh OANDA H1 candles (bid/ask OHLC) into DATA_DIR/*.csv.

Writes the exact schema the trainer reads (read_pair_csv):
    time,bid_o,bid_h,bid_l,bid_c,ask_o,ask_h,ask_l,ask_c

Stdlib only (urllib) — no extra installs. Reads OANDA_TOKEN / OANDA_BASE_URL from the
environment or the local .env. Run in YOUR environment (the sandbox can't reach OANDA).

    python tools/download_oanda_candles.py                    # all pairs, ~12000 H1 bars
    python tools/download_oanda_candles.py --count 20000
    python tools/download_oanda_candles.py --pairs EUR_USD USD_JPY --granularity H1

Then retrain:
    python 03_train_h1_auto_models.py      # honest trainer (purge + OOS gate + horizon_end)
    python create_h1_registry.py           # rebuild registry (real avg_auc + tradable bar)
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", REPO / "oanda_h1_ba_live"))
MAX_PER_REQ = 5000   # OANDA hard limit per candles request


def load_env():
    env = dict(os.environ)
    dotenv = REPO / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    return env


def default_instruments():
    """Instrument codes from the existing CSVs (EURUSD -> EUR_USD)."""
    out = []
    for p in sorted(DATA_DIR.glob("*.csv")):
        s = p.stem.upper()
        out.append(s if "_" in s else f"{s[:3]}_{s[3:]}")
    return out or ["EUR_USD", "USD_JPY", "GBP_USD"]


def fetch_chunk(base, token, instrument, granularity, count, to_iso=None):
    params = {"granularity": granularity, "price": "BA", "count": str(count)}
    if to_iso:
        params["to"] = to_iso
    url = f"{base}/v3/instruments/{instrument}/candles?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r).get("candles", [])


def download_instrument(base, token, instrument, granularity, target):
    """Paginate backwards until `target` complete candles collected."""
    collected = {}
    to_iso = None
    while len(collected) < target:
        want = min(MAX_PER_REQ, target - len(collected) + 5)
        try:
            candles = fetch_chunk(base, token, instrument, granularity, want, to_iso)
        except Exception as e:
            print(f"    {instrument}: request failed ({repr(e)[:120]}); stopping.")
            break
        if not candles:
            break
        new = 0
        for c in candles:
            if not c.get("complete"):
                continue
            t = c["time"]
            if t in collected:
                continue
            collected[t] = c
            new += 1
        earliest = candles[0]["time"]
        if to_iso == earliest or new == 0:
            break   # no further history
        to_iso = earliest
        time.sleep(0.15)   # be gentle with the API
    return [collected[t] for t in sorted(collected)]


def write_csv(path, candles):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "bid_o", "bid_h", "bid_l", "bid_c",
                    "ask_o", "ask_h", "ask_l", "ask_c"])
        for c in candles:
            b, a = c["bid"], c["ask"]
            w.writerow([c["time"], b["o"], b["h"], b["l"], b["c"],
                        a["o"], a["h"], a["l"], a["c"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", nargs="*", help="instruments like EUR_USD (default: existing CSVs)")
    ap.add_argument("--count", type=int, default=12000, help="target candles per pair")
    ap.add_argument("--granularity", default="H1")
    args = ap.parse_args()

    env = load_env()
    token = env.get("OANDA_TOKEN", "").strip()
    base = env.get("OANDA_BASE_URL", "https://api-fxtrade.oanda.com").strip().rstrip("/")
    if not token:
        sys.exit("OANDA_TOKEN not set (env or .env). Aborting.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    instruments = args.pairs or default_instruments()
    print(f"Downloading {args.granularity} candles for {len(instruments)} pair(s) -> {DATA_DIR}")
    print(f"base={base}  target={args.count}/pair\n")

    for inst in instruments:
        candles = download_instrument(base, token, inst, args.granularity, args.count)
        if not candles:
            print(f"  {inst:8}  NO DATA")
            continue
        out = DATA_DIR / f"{inst.replace('_','')}.csv"
        write_csv(out, candles)
        span = f"{candles[0]['time'][:10]} .. {candles[-1]['time'][:10]}"
        print(f"  {inst:8}  {len(candles):6d} bars  {span}  -> {out.name}")

    print("\nDone. Next:")
    print("  python 03_train_h1_auto_models.py")
    print("  python create_h1_registry.py")


if __name__ == "__main__":
    main()
