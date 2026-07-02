#!/usr/bin/env python3
"""Download/update OANDA H1 candles (bid/ask OHLC) into DATA_DIR/*.csv.

Writes the exact schema the trainer reads (read_pair_csv):
    time,bid_o,bid_h,bid_l,bid_c,ask_o,ask_h,ask_l,ask_c

Stdlib only (urllib) — no extra installs. Reads OANDA_TOKEN / OANDA_BASE_URL from the
environment or the local .env. Run in YOUR environment (the sandbox can't reach OANDA).

    python tools/download_oanda_candles.py                    # update existing CSVs
    python tools/download_oanda_candles.py --replace --count 20000
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


FIELDNAMES = ["time", "bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]


def canonical_time(value):
    text = str(value or "").strip()
    if not text:
        return ""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:
        head, tail = text.split(".", 1)
        tz = "+00:00" if tail.endswith("+00:00") else ""
        frac = tail.replace("+00:00", "")[:6].ljust(6, "0")
        text = f"{head}.{frac}{tz}"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return str(value).strip()
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_chunk(base, token, instrument, granularity, count, to_iso=None, from_iso=None):
    params = {"granularity": granularity, "price": "BA", "count": str(count)}
    if to_iso:
        params["to"] = to_iso
    if from_iso:
        params["from"] = from_iso
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


def update_instrument(base, token, instrument, granularity, since_iso):
    """Download complete candles newer than the latest local timestamp."""
    collected = {}
    from_iso = since_iso
    while True:
        try:
            candles = fetch_chunk(base, token, instrument, granularity, MAX_PER_REQ, from_iso=from_iso)
        except Exception as e:
            print(f"    {instrument}: request failed ({repr(e)[:120]}); stopping.")
            break
        complete = [c for c in candles if c.get("complete")]
        new = 0
        for c in complete:
            t = canonical_time(c["time"])
            if t <= since_iso or t in collected:
                continue
            collected[t] = c
            new += 1
        if len(candles) < MAX_PER_REQ or new == 0:
            break
        from_iso = sorted(collected)[-1]
        time.sleep(0.15)
    return [collected[t] for t in sorted(collected)]


def candle_to_row(c):
    b, a = c["bid"], c["ask"]
    return {
        "time": canonical_time(c["time"]),
        "bid_o": b["o"], "bid_h": b["h"], "bid_l": b["l"], "bid_c": b["c"],
        "ask_o": a["o"], "ask_h": a["h"], "ask_l": a["l"], "ask_c": a["c"],
    }


def read_existing_rows(path):
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = {}
        for row in csv.DictReader(f):
            t = canonical_time(row.get("time"))
            if t:
                rows[t] = {k: (t if k == "time" else row.get(k, "")) for k in FIELDNAMES}
        return rows


def write_rows(path, rows_by_time):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for t in sorted(rows_by_time):
            w.writerow(rows_by_time[t])


def write_csv(path, candles):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDNAMES)
        for c in candles:
            row = candle_to_row(c)
            w.writerow([row[k] for k in FIELDNAMES])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", nargs="*", help="instruments like EUR_USD (default: existing CSVs)")
    ap.add_argument("--count", type=int, default=12000, help="target candles per pair")
    ap.add_argument("--granularity", default="H1")
    ap.add_argument("--replace", action="store_true", help="rewrite each CSV with the latest --count candles")
    args = ap.parse_args()

    env = load_env()
    token = env.get("OANDA_TOKEN", "").strip()
    base = env.get("OANDA_BASE_URL", "https://api-fxtrade.oanda.com").strip().rstrip("/")
    if not token:
        sys.exit("OANDA_TOKEN not set (env or .env). Aborting.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    instruments = args.pairs or default_instruments()
    mode = "replace" if args.replace else "update"
    print(f"{mode.title()} {args.granularity} candles for {len(instruments)} pair(s) -> {DATA_DIR}")
    print(f"base={base}  target={args.count}/pair\n")

    for inst in instruments:
        out = DATA_DIR / f"{inst.replace('_','')}.csv"
        if args.replace or not out.exists():
            candles = download_instrument(base, token, inst, args.granularity, args.count)
            if not candles:
                print(f"  {inst:8}  NO DATA")
                continue
            write_csv(out, candles)
            span = f"{candles[0]['time'][:10]} .. {candles[-1]['time'][:10]}"
            print(f"  {inst:8}  {len(candles):6d} bars  {span}  -> {out.name}")
            continue

        rows = read_existing_rows(out)
        latest = max(rows) if rows else None
        if not latest:
            candles = download_instrument(base, token, inst, args.granularity, args.count)
            if not candles:
                print(f"  {inst:8}  NO DATA")
                continue
            rows = {c["time"]: candle_to_row(c) for c in candles}
            added = len(rows)
        else:
            candles = update_instrument(base, token, inst, args.granularity, latest)
            for c in candles:
                rows[c["time"]] = candle_to_row(c)
            added = len(candles)
        write_rows(out, rows)
        first, last = min(rows), max(rows)
        print(f"  {inst:8}  +{added:5d} bars  total={len(rows):6d}  {first[:10]} .. {last[:10]}  -> {out.name}")

    print("\nDone. Next:")
    print("  python 03_train_h1_auto_models.py")
    print("  python create_h1_registry.py")


if __name__ == "__main__":
    main()
