# -*- coding: utf-8 -*-
# Stock Signals PRO – v2 (backend overhaul, UI kept minimal and compatible)
# Date: 2025-08-14
# Notes:
# - Conservative multi-asset rules with risk-parity, trend/momentum/low-vol scoring,
#   monthly rebalance, volatility targeting, costs & slippage, walk-forward OOS.
# - No change to visual design sections; content is computed differently "under the hood".

import os, json, math, datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

APP_TITLE = "📈 Stock Signals PRO – Enhanced Backend v2"

# ----------------------- Utils -----------------------
def annualize_return(daily_ret: pd.Series) -> float:
    mu = daily_ret.mean()
    return (1 + mu)**252 - 1

def sharpe_ratio(daily_ret: pd.Series, rf: float = 0.0) -> float:
    # rf is daily risk-free rate; assume 0 for simplicity
    ex = daily_ret - rf
    vol = ex.std()
    if vol == 0 or np.isnan(vol):
        return 0.0
    return (ex.mean() / vol) * np.sqrt(252)

def max_drawdown(cum_ret: pd.Series) -> float:
    roll_max = cum_ret.cummax()
    dd = cum_ret / roll_max - 1.0
    return dd.min()

def turnover_from_weights(weights: pd.DataFrame) -> float:
    # weights: index=rebal dates, columns=tickers
    w = weights.fillna(0.0).copy()
    tw = (w.diff().abs().sum(axis=1)).dropna()
    # Periodic turnover (sum abs change). Annualize assuming monthly rebal if monthly index
    if len(w.index) < 2:
        return 0.0
    # infer periods per year
    idx = w.index.to_series().diff().dropna().dt.days
    avg_days = idx.mean() if len(idx) else 30
    periods_per_year = max(1, int(round(365/avg_days)))
    return tw.mean() * periods_per_year

def end_of_month_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    eom = prices.resample("M").last().index
    return eom[eom.isin(prices.index)]

def vol_target_scale(port_ret_daily: pd.Series, target_ann_vol: float) -> float:
    # Estimate realized daily vol and compute scaling factor
    if len(port_ret_daily) < 20:
        return 1.0
    realized = port_ret_daily.std() * np.sqrt(252)
    if realized == 0 or np.isnan(realized):
        return 1.0
    return np.clip(target_ann_vol / realized, 0.0, 1.0)  # cap leverage at 1x

# ----------------------- Data -----------------------
def fetch_history(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how="all")
    data = data.ffill().dropna()
    return data

# ----------------------- Scoring -----------------------
@dataclass
class ScoreWeights:
    momentum: float = 0.5
    trend: float = 0.3
    low_vol: float = 0.2

def compute_scores(pr: pd.DataFrame, mom_lookback=126, trend_ma=200, vol_lookback=20,
                   weights: ScoreWeights = ScoreWeights()) -> pd.DataFrame:
    rets = pr.pct_change()
    mom = (pr / pr.shift(mom_lookback) - 1.0).clip(lower=-1, upper=2)
    ma = pr.rolling(trend_ma).mean()
    trend = (pr > ma).astype(float) * 2 - 1  # +1 if above MA, -1 if below
    vol = rets.rolling(vol_lookback).std()
    low_vol = (-vol).rank(axis=1, pct=True) * 2 - 1  # higher is better
    
    # z-score each component to balance scales
    def zscore(df):
        return (df - df.mean()) / (df.std() + 1e-9)
    mom_z = zscore(mom)
    trend_z = trend  # already -1..+1
    low_vol_z = zscore(low_vol)
    
    score = (weights.momentum * mom_z) + (weights.trend * trend_z) + (weights.low_vol * low_vol_z)
    return score

def risk_parity_weights(pr: pd.DataFrame, lookback=20) -> pd.Series:
    vol = pr.pct_change().rolling(lookback).std().iloc[-1].replace(0, np.nan)
    inv = 1.0 / vol
    inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if inv.sum() == 0:
        return pd.Series(0, index=pr.columns)
    w = inv / inv.sum()
    return w

# ----------------------- Strategy -----------------------
@dataclass
class StratCfg:
    top_k: int = 3
    rebalance: str = "M"
    mom_lb: int = 126
    trend_ma: int = 200
    vol_lb: int = 20
    target_vol_annual: float = 0.10
    max_gross_leverage: float = 1.0
    cost_bp: int = 10
    slip_bp: int = 5
    cash_ticker: str = "SHY"
    score_weights: ScoreWeights = ScoreWeights()

def run_strategy(prices: pd.DataFrame, risk_on: List[str], defensive: List[str], cfg: StratCfg) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    prices = prices.copy()
    rebals = end_of_month_dates(prices)
    scores = compute_scores(prices, cfg.mom_lb, cfg.trend_ma, cfg.vol_lb, cfg.score_weights)
    weights_list = []
    port_daily_ret = pd.Series(index=prices.index, dtype=float)
    
    # Determine risk regime using SPY 200MA; if SPY below MA -> defensive
    spy = prices["SPY"] if "SPY" in prices.columns else prices.iloc[:,0]
    spy_ma = spy.rolling(cfg.trend_ma).mean()
    risk_on_regime = spy > spy_ma
    
    prev_w = pd.Series(0, index=prices.columns, dtype=float)
    for d in rebals:
        if d not in prices.index: 
            continue
        regime_on = bool(risk_on_regime.loc[:d].iloc[-1])
        pool = risk_on if regime_on else defensive
        pool = [t for t in pool if t in prices.columns]
        if len(pool) == 0:
            continue
        # Select top-K by score
        sc = scores.loc[d, pool].sort_values(ascending=False)
        picks = list(sc.iloc[:cfg.top_k].index)
        # Risk parity among picks (fallback equal weight)
        rp_w = risk_parity_weights(prices[picks], cfg.vol_lb)
        if rp_w.sum() == 0 or rp_w.isna().all():
            rp_w = pd.Series(1/len(picks), index=picks)
        w = pd.Series(0.0, index=prices.columns)
        w.loc[picks] = rp_w.values
        # Cap gross leverage and ensure cash uses remainder (if any)
        gross = float(w.abs().sum())
        if gross > cfg.max_gross_leverage:
            w *= cfg.max_gross_leverage / gross
        if cfg.cash_ticker in prices.columns:
            cash_w = max(0.0, 1.0 - w.sum())
            w[cfg.cash_ticker] = w.get(cfg.cash_ticker, 0.0) + cash_w
        weights_list.append((d, w))
        
        # Apply daily returns until next rebalance date
        if d == rebals[-1]:
            next_d = prices.index[-1]
        else:
            next_d = rebals[rebals.get_loc(d)+1]
        window = prices.loc[d:next_d]
        rets = window.pct_change().fillna(0.0)
        # Apply trading costs on rebalance day proportional to weight change
        turnover = float((w - prev_w).abs().sum())
        trade_cost = turnover * (cfg.cost_bp + cfg.slip_bp) / 10000.0
        if len(rets) > 0:
            # subtract cost on first day after rebalance
            first = rets.index[0]
            if first in port_daily_ret.index:
                port_daily_ret.loc[first] = 0.0  # ensure exists
            # daily portfolio return = sum(w * asset return)
            prt = (rets * w).sum(axis=1)
            if len(prt) > 0:
                prt.iloc[0] = prt.iloc[0] - trade_cost
                port_daily_ret.loc[prt.index] = prt.values
        prev_w = w
    
    weights_df = pd.DataFrame({d: w for d, w in weights_list}).T
    weights_df.index.name = "RebalanceDate"
    port_daily_ret = port_daily_ret.fillna(0.0)
    cum = (1 + port_daily_ret).cumprod()
    return port_daily_ret, cum, weights_df

# ----------------------- Walk-Forward -----------------------
def walk_forward(pr: pd.DataFrame, risk_on: List[str], defensive: List[str], cfg: StratCfg,
                 train_months=36, test_months=12) -> Dict[str, object]:
    start = pr.index.min()
    end = pr.index.max()
    # Build rolling windows
    oos_returns = []
    oos_weights = []
    cursor = pr.index[0] + pd.offsets.DateOffset(months=train_months)
    while cursor + pd.offsets.DateOffset(months=test_months) < end:
        train_end = cursor
        test_end = cursor + pd.offsets.DateOffset(months=test_months)
        train = pr.loc[:train_end]
        test = pr.loc[train_end:test_end]
        if len(train) < 250 or len(test) < 20:
            cursor += pd.offsets.DateOffset(months=test_months)
            continue
        # Run on full data but only collect test period returns
        ret_daily, cum, wdf = run_strategy(pr.loc[:test_end], risk_on, defensive, cfg)
        test_ret = ret_daily.loc[test.index]
        oos_returns.append(test_ret)
        oos_weights.append(wdf.loc[wdf.index.intersection(test.index)])
        cursor += pd.offsets.DateOffset(months=test_months)
    if len(oos_returns) == 0:
        return {"oos_daily": pd.Series(dtype=float), "oos_weights": pd.DataFrame()}
    oos_daily = pd.concat(oos_returns).sort_index()
    oos_weights_df = pd.concat(oos_weights).sort_index() if len(oos_weights) else pd.DataFrame()
    return {"oos_daily": oos_daily, "oos_weights": oos_weights_df}

def metrics_from_daily(daily: pd.Series, weights_df: pd.DataFrame=None) -> Dict[str, float]:
    daily = daily.dropna()
    if len(daily) == 0:
        return {"CAGR": 0.0, "MaxDD": 0.0, "Sharpe": 0.0, "turnover": 0.0}
    cum = (1+daily).cumprod()
    start, end = cum.index[0], cum.index[-1]
    years = max(1e-9, (end - start).days / 365.25)
    cagr = cum.iloc[-1]**(1/years) - 1 if cum.iloc[-1] > 0 else -1.0
    mdd = max_drawdown(cum)
    shp = sharpe_ratio(daily)
    to = turnover_from_weights(weights_df) if weights_df is not None and not weights_df.empty else 0.0
    return {"CAGR": float(cagr), "MaxDD": float(mdd), "Sharpe": float(shp), "turnover": float(to)}

# ----------------------- Streamlit App -----------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Backend: risk-parity + momentum/trend/low-vol, monthly rebalance, vol targeting, costs & walk-forward OOS.")
    
    # Load config
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    uni = cfg["universe"]
    strat_cfg = StratCfg(
        top_k = cfg["strategy"]["top_k"],
        rebalance = cfg["strategy"]["rebalance"],
        mom_lb = cfg["strategy"]["mom_lookback_days"],
        trend_ma = cfg["strategy"]["trend_ma_days"],
        vol_lb = cfg["strategy"]["vol_lookback_days"],
        target_vol_annual = cfg["strategy"]["target_vol_annual"],
        max_gross_leverage = cfg["strategy"]["max_gross_leverage"],
        cost_bp = cfg["strategy"]["txn_cost_bp_per_trade"],
        slip_bp = cfg["strategy"]["slippage_bp_per_trade"],
        cash_ticker = cfg["strategy"]["cash_ticker"],
        score_weights = ScoreWeights(**cfg["strategy"]["score_weights"]),
    )
    
    # Sidebar controls (non-visual change; just parameters)
    with st.sidebar:
        st.header("Settings")
        start_date = st.date_input("Start date", value=dt.date.today() - dt.timedelta(days=365*10))
        end_date = st.date_input("End date", value=dt.date.today())
        top_k = st.slider("Top K", 1, 6, strat_cfg.top_k)
        target_vol = st.slider("Target Vol (ann.)", 0.05, 0.25, float(strat_cfg.target_vol_annual), 0.01)
        cost_bp = st.slider("Cost (bp/trade)", 0, 50, strat_cfg.cost_bp, 1)
        slip_bp = st.slider("Slippage (bp/trade)", 0, 50, strat_cfg.slip_bp, 1)
        st.caption("UI is kept simple; parameters only affect backend calculations.")
    
    strat_cfg.top_k = top_k
    strat_cfg.target_vol_annual = target_vol
    strat_cfg.cost_bp = cost_bp
    strat_cfg.slip_bp = slip_bp
    
    tickers = sorted(set(uni["risk_on"] + uni["defensive"]))
    pr = fetch_history(tickers, start=start_date.isoformat(), end=(end_date + dt.timedelta(days=1)).isoformat())
    if pr.empty:
        st.error("No price data fetched. Check internet or tickers.")
        return
    
    # Run walk-forward OOS
    wf = walk_forward(pr, uni["risk_on"], uni["defensive"], strat_cfg,
                      train_months=cfg["strategy"]["train_months"],
                      test_months=cfg["strategy"]["test_months"])
    oos_daily = wf["oos_daily"]
    oos_w = wf["oos_weights"]
    
    # Vol targeting applied after the fact (scale daily returns to target)
    scale = vol_target_scale(oos_daily, strat_cfg.target_vol_annual)
    oos_daily_scaled = oos_daily * scale
    mets = metrics_from_daily(oos_daily_scaled, oos_w)
    
    # Display OOS metrics
    st.subheader("Portfolio OOS Metrics (Walk-Forward)")
    st.write(f"**CAGR:** {mets['CAGR']:.2%} · **MaxDD:** {mets['MaxDD']:.2%} · **Sharpe:** {mets['Sharpe']:.2f} · **turnover:** {mets['turnover']:.2f}")
    
    # Cumulative chart
    import plotly.express as px
    cum = (1 + oos_daily_scaled).cumprod()
    fig = px.line(cum, title="Cumulative OOS (scaled to target vol)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Current Suggested Allocation")
    if not oos_w.empty:
        last_w = oos_w.iloc[-1].sort_values(ascending=False)
        st.dataframe(last_w.to_frame("Weight"))
    else:
        st.info("No OOS weights available yet (insufficient history).")
    
    # Chat-like instruction (simple, to preserve feel)
    st.subheader("Chat Instructions")
    prompt = st.text_input("Ask about the portfolio or a ticker:")
    if prompt:
        # Very simple rules-based response based on last weights & regime
        ans = []
        if not oos_w.empty:
            top = last_w[last_w > 0.01].index.tolist()
            if top:
                ans.append("Top holdings now: " + ", ".join(top[:5]))
            else:
                ans.append("Currently holding mostly cash/short-duration (defensive posture).")
        ans.append("This system uses trend + momentum + low-vol scoring with monthly rebalances, risk parity sizing, and volatility targeting.")
        ans.append("Use the metrics above as OOS guidance; they include trading costs and slippage.")
        st.success(" ".join(ans))

if __name__ == "__main__":
    main()
