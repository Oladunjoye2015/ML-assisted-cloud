#!/usr/bin/env python3
"""
Honest walk-forward evaluation PREVIEW on real CSV data.

Numpy-only. Uses a faithful logistic-regression proxy of the deployed
LogisticRegression pipeline (median impute -> standardize -> L2, balanced classes),
the SAME features and SAME ATR triple-barrier labels as 03_train_h1_auto_models.py,
but adds the three methodology fixes the current trainer lacks:

  * purge: drop the last HORIZON_BARS training rows whose labels look into the test window
  * out-of-sample gate: pick conf/margin gate on TRAIN, apply to TEST (no in-sample peeking)
  * cost-aware P&L with bootstrap 95% CI (win=+TP_ATR, loss=-SL_ATR, minus spread)

This is a preview of edge for the linear model family (the deployed 'best' for most
pairs). The production evaluator (training/evaluate.py) runs the full model set in the
user's venv.
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import pandas as pd

# --- config mirrors the training script defaults ---------------------------
HORIZON_BARS = 8
TP_ATR = 1.3
SL_ATR = 1.0
N_FOLDS = 5
TEST_FRACTION = 0.40          # last 40% used for walk-forward testing
MIN_GATE_TRADES = 30
GATE_GRID = [0.52,0.54,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70]
MARGIN_GRID = [0.02,0.04,0.06,0.08,0.10,0.12]
COST_SPREAD_MULT = 1.0        # round-trip cost = this * spread (in ATR units)
RNG = np.random.default_rng(42)
DATA_DIR = os.getenv("DATA_DIR", "/sessions/upbeat-exciting-edison/mnt/ML-assisted-cloud/oanda_h1_ba_live")

def pip_size_from_pair(pair):
    return 0.01 if pair.endswith("JPY") else 0.0001

def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def rsi(c, period=14):
    d = c.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    rs = up.ewm(alpha=1/period, adjust=False).mean() / dn.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)
    return 100 - 100/(1+rs)

def atr(df, period=14):
    h,l,c = df["mid_h"], df["mid_l"], df["mid_c"]; pc = c.shift(1)
    tr = pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(df, period=14):
    h,l,c = df["mid_h"],df["mid_l"],df["mid_c"]; pc=c.shift(1)
    up=h.diff(); dn=-l.diff()
    plus=np.where((up>dn)&(up>0),up,0.0); minus=np.where((dn>up)&(dn>0),dn,0.0)
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr_=tr.ewm(alpha=1/period,adjust=False).mean().replace(0,np.nan)
    pdi=100*pd.Series(plus,index=df.index).ewm(alpha=1/period,adjust=False).mean()/atr_
    mdi=100*pd.Series(minus,index=df.index).ewm(alpha=1/period,adjust=False).mean()/atr_
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/period,adjust=False).mean()

def read_pair_csv(path):
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    has_mid = all(c in df.columns for c in ["mid_o","mid_h","mid_l","mid_c"])
    has_ba = all(c in df.columns for c in ["bid_o","bid_h","bid_l","bid_c","ask_o","ask_h","ask_l","ask_c"])
    if not has_mid:
        for x in ["o","h","l","c"]:
            df[f"mid_{x}"]=(df[f"bid_{x}"]+df[f"ask_{x}"])/2.0
    if "spread_c" not in df.columns and has_ba:
        df["spread_c"]=df["ask_c"]-df["bid_c"]
    if "volume" not in df.columns: df["volume"]=0
    return df

def add_features(df, pair):
    df=df.copy(); ps=pip_size_from_pair(pair); c=df["mid_c"]
    for k in (1,2,3,6,12,24): df[f"ret{k}"]=c.pct_change(k)
    df["range_pips"]=(df["mid_h"]-df["mid_l"])/ps
    df["body_pips"]=(df["mid_c"]-df["mid_o"])/ps
    df["upper_wick_pips"]=(df["mid_h"]-df[["mid_o","mid_c"]].max(axis=1))/ps
    df["lower_wick_pips"]=(df[["mid_o","mid_c"]].min(axis=1)-df["mid_l"])/ps
    for sp in (20,50,100,200): df[f"ema{sp}"]=ema(c,sp)
    for sp in (20,50,100,200): df[f"dist_ema{sp}_pips"]=(c-df[f"ema{sp}"])/ps
    df["ema20_slope"]=df["ema20"].diff(3)/ps
    df["ema50_slope"]=df["ema50"].diff(6)/ps
    df["ema200_slope"]=df["ema200"].diff(12)/ps
    df["rsi14"]=rsi(c,14); df["rsi7"]=rsi(c,7)
    df["atr14"]=atr(df,14); df["atr14_pips"]=df["atr14"]/ps
    df["adx14"]=adx(df,14)
    e12,e26=ema(c,12),ema(c,26); df["macd"]=e12-e26
    df["macd_signal"]=ema(df["macd"],9); df["macdh"]=df["macd"]-df["macd_signal"]
    df["macdh_pips"]=df["macdh"]/ps
    r20=c.rolling(20); df["bb_mid"]=r20.mean(); df["bb_std"]=r20.std()
    df["bb_upper"]=df["bb_mid"]+2*df["bb_std"]; df["bb_lower"]=df["bb_mid"]-2*df["bb_std"]
    df["bb_width_pips"]=(df["bb_upper"]-df["bb_lower"])/ps
    df["bb_pos"]=(c-df["bb_lower"])/(df["bb_upper"]-df["bb_lower"]).replace(0,np.nan)
    df["spread_pips"]=df["spread_c"]/ps if "spread_c" in df.columns else 0.0
    df["spread_atr"]=df["spread_pips"]/df["atr14_pips"].replace(0,np.nan)
    if "time" in df.columns:
        t=pd.to_datetime(df["time"],utc=True,errors="coerce")
        df["hour_utc"]=t.dt.hour; df["day_of_week"]=t.dt.dayofweek; df["month"]=t.dt.month
        df["hour_sin"]=np.sin(2*np.pi*df["hour_utc"]/24); df["hour_cos"]=np.cos(2*np.pi*df["hour_utc"]/24)
        df["dow_sin"]=np.sin(2*np.pi*df["day_of_week"]/7); df["dow_cos"]=np.cos(2*np.pi*df["day_of_week"]/7)
    df["trend_up"]=(df["ema20"]>df["ema50"]).astype(int)
    df["trend_down"]=(df["ema20"]<df["ema50"]).astype(int)
    df["price_above_ema200"]=(c>df["ema200"]).astype(int)
    fcols=["ret1","ret2","ret3","ret6","ret12","ret24","range_pips","body_pips","upper_wick_pips",
        "lower_wick_pips","dist_ema20_pips","dist_ema50_pips","dist_ema100_pips","dist_ema200_pips",
        "ema20_slope","ema50_slope","ema200_slope","rsi14","rsi7","atr14_pips","adx14","macdh_pips",
        "bb_width_pips","bb_pos","spread_pips","spread_atr","hour_utc","day_of_week","month",
        "hour_sin","hour_cos","dow_sin","dow_cos","trend_up","trend_down","price_above_ema200","volume"]
    return df, fcols

def build_labels(df):
    df=df.copy(); n=len(df)
    y=np.full(n,np.nan); H=highs=df["mid_h"].values; L=df["mid_l"].values
    C=df["mid_c"].values; A=df["atr14"].values
    for i in range(n-HORIZON_BARS-1):
        e=C[i]; a=A[i]
        if not np.isfinite(e) or not np.isfinite(a) or a<=0: continue
        ltp=e+TP_ATR*a; lsl=e-SL_ATR*a; stp=e-TP_ATR*a; ssl=e+SL_ATR*a
        for j in range(1,HORIZON_BARS+1):
            h=H[i+j]; l=L[i+j]
            long_tp=h>=ltp; long_sl=l<=lsl; short_tp=l<=stp; short_sl=h>=ssl
            if long_tp and long_sl: break
            if short_tp and short_sl: break
            if long_tp and not short_tp: y[i]=1; break
            if short_tp and not long_tp: y[i]=0; break
            if long_sl and short_sl: break
    df["y"]=y
    return df

def fit_logreg(X, y, l2=2.0, iters=300, lr=0.5):
    # standardized inputs assumed; balanced class weights; gradient descent
    n,d = X.shape
    w=np.zeros(d); b=0.0
    pos=max(1,(y==1).sum()); neg=max(1,(y==0).sum())
    wpos=n/(2*pos); wneg=n/(2*neg)
    sw=np.where(y==1,wpos,wneg)
    for _ in range(iters):
        z=X@w+b; p=1/(1+np.exp(-np.clip(z,-30,30)))
        g=(p-y)*sw
        gw=X.T@g/n + l2*w/n; gb=g.mean()
        w-=lr*gw; b-=lr*gb
    return w,b

def predict_proba(X,w,b):
    return 1/(1+np.exp(-np.clip(X@w+b,-30,30)))

def standardize_fit(Xtr):
    med=np.nanmedian(Xtr,axis=0)
    Xtr=np.where(np.isnan(Xtr),med,Xtr)
    mu=Xtr.mean(axis=0); sd=Xtr.std(axis=0); sd[sd==0]=1.0
    return med,mu,sd
def standardize_apply(X,med,mu,sd):
    X=np.where(np.isnan(X),med,X)
    return (X-mu)/sd

def choose_gate(conf, margin, correct):
    best=(0.0,0.56,0.06,0)
    for g in GATE_GRID:
        for m in MARGIN_GRID:
            mask=(conf>=g)&(margin>=m); t=int(mask.sum())
            if t<MIN_GATE_TRADES: continue
            prec=float(correct[mask].mean())
            if prec>best[0] or (abs(prec-best[0])<1e-9 and t>best[3]):
                best=(prec,g,m,t)
    return best[1],best[2]

def evaluate_pair(path):
    pair=os.path.basename(path).split(".")[0].upper()
    df=read_pair_csv(path); df,fcols=add_features(df,pair); df=build_labels(df)
    df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=fcols+["y"]).reset_index(drop=True)
    if len(df)<1500: return {"pair":pair,"ok":False,"reason":f"rows={len(df)}"}
    X=df[fcols].values.astype(float); y=df["y"].astype(int).values
    spread_atr=df["spread_atr"].values.astype(float)
    n=len(df); test_start=int(n*(1-TEST_FRACTION)); test_len=(n-test_start)//N_FOLDS
    if test_len<80: return {"pair":pair,"ok":False,"reason":"too_few_test"}
    oos_p=[]; oos_y=[]; trade_pnl=[]; trade_correct=[]
    for k in range(N_FOLDS):
        ts=test_start+k*test_len; te=ts+test_len if k<N_FOLDS-1 else n
        tr_end=ts-HORIZON_BARS  # PURGE
        if tr_end<300: continue
        Xtr,ytr=X[:tr_end],y[:tr_end]
        if len(np.unique(ytr))<2: continue
        med,mu,sd=standardize_fit(Xtr)
        Xtr_s=standardize_apply(Xtr,med,mu,sd)
        w,b=fit_logreg(Xtr_s,ytr)
        # OOS gate: pick on train predictions, apply to test
        ptr=predict_proba(Xtr_s,w,b)
        ctr=((ptr>=0.5).astype(int)==ytr).astype(int)
        conf_tr=np.maximum(ptr,1-ptr); marg_tr=np.abs(ptr-0.5)*2
        g,m=choose_gate(conf_tr,marg_tr,ctr)
        Xte_s=standardize_apply(X[ts:te],med,mu,sd)
        pte=predict_proba(Xte_s,w,b); yte=y[ts:te]; sate=spread_atr[ts:te]
        oos_p.append(pte); oos_y.append(yte)
        conf_te=np.maximum(pte,1-pte); marg_te=np.abs(pte-0.5)*2
        pred=(pte>=0.5).astype(int); corr=(pred==yte).astype(int)
        mask=(conf_te>=g)&(marg_te>=m)
        for idx in np.where(mask)[0]:
            win=corr[idx]==1
            cost=COST_SPREAD_MULT*(sate[idx] if np.isfinite(sate[idx]) else 0.0)
            pnl=(TP_ATR if win else -SL_ATR)-cost
            trade_pnl.append(pnl); trade_correct.append(int(win))
    if not oos_p: return {"pair":pair,"ok":False,"reason":"no_folds"}
    P=np.concatenate(oos_p); Y=np.concatenate(oos_y)
    auc=auc_score(Y,P)
    tp=np.array(trade_pnl); tc=np.array(trade_correct)
    ntr=len(tp)
    if ntr>=10:
        exp=float(tp.mean()); prec=float(tc.mean())
        boot=[RNG.choice(tp,size=ntr,replace=True).mean() for _ in range(2000)]
        lo,hi=np.percentile(boot,[2.5,97.5])
    else:
        exp=prec=lo=hi=float("nan")
    return {"pair":pair,"ok":True,"rows":len(df),"oos_rows":len(Y),"oos_auc":round(float(auc),4),
        "n_trades":ntr,"precision":round(prec,4) if ntr>=10 else None,
        "expectancy_R":round(exp,4) if ntr>=10 else None,
        "exp_ci_lo":round(float(lo),4) if ntr>=10 else None,
        "exp_ci_hi":round(float(hi),4) if ntr>=10 else None,
        "neutral_dropped_pct":round(100*(1-len(df)/max(1,len(read_pair_csv(path)))),1)}

def auc_score(y,p):
    y=np.asarray(y); p=np.asarray(p)
    n1=(y==1).sum(); n0=(y==0).sum()
    if n1==0 or n0==0: return float("nan")
    order=np.argsort(p); ranks=np.empty(len(p)); ranks[order]=np.arange(1,len(p)+1)
    # average ties
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def main():
    rows=[]
    for path in sorted(glob.glob(os.path.join(DATA_DIR,"*.csv"))):
        try:
            r=evaluate_pair(path)
        except Exception as e:
            r={"pair":os.path.basename(path),"ok":False,"reason":repr(e)}
        rows.append(r); print(json.dumps(r))
    out=os.path.join(os.path.dirname(__file__),"eval_preview_results.json")
    json.dump(rows,open(out,"w"),indent=2)
    print("WROTE",out)

if __name__=="__main__":
    main()
