"""
Macro-level features:
1. Northbound/Southbound Stock Connect net flow (via akshare, East Money source)
2. GARCH(1,1) conditional volatility (via arch package)
3. ADR-implied overnight proxy for key dual-listed HK stocks

All functions fail-safe: return neutral defaults (0.0) on any error so the
main pipeline never breaks due to this supplementary data.
"""

import numpy as np
import pandas as pd
from config import ADR_PROXIES


# ---------------------------------------------------------------------------
# 1. Northbound / Southbound flow
# ---------------------------------------------------------------------------

def get_stock_connect_flow() -> dict:
    """
    Returns dict with latest northbound (into A-shares) and southbound
    (into HK stocks) net flow in RMB/HKD 100 millions.
    Southbound flow is the more relevant leading indicator for HSI.
    """
    try:
        import akshare as ak
        df = ak.stock_hsgt_hist_em(symbol="南向资金")  # Southbound historical net flow
        df = df.sort_values("日期")
        latest = df.iloc[-1]
        prev5 = df.iloc[-6:-1]["当日成交净买额"].astype(float).mean()

        return {
            "southbound_net_flow_latest": float(latest["当日成交净买额"]),
            "southbound_net_flow_5d_avg": float(prev5),
            "southbound_flow_accelerating": float(latest["当日成交净买额"]) > float(prev5),
        }
    except Exception as e:
        print(f"Stock Connect flow fetch failed (using neutral defaults): {e}")
        return {
            "southbound_net_flow_latest": 0.0,
            "southbound_net_flow_5d_avg": 0.0,
            "southbound_flow_accelerating": False,
        }


# ---------------------------------------------------------------------------
# 2. GARCH(1,1) conditional volatility
# ---------------------------------------------------------------------------

def compute_garch_volatility(close_prices: pd.Series) -> pd.Series:
    """
    Fits GARCH(1,1) on log returns and returns the conditional volatility series,
    aligned to the original index. Captures volatility clustering that simple
    rolling-window ATR cannot.
    """
    try:
        from arch import arch_model

        returns = np.log(close_prices / close_prices.shift(1)).dropna() * 100
        if len(returns) < 100:
            raise ValueError("Insufficient data for GARCH fitting (<100 obs)")

        model = arch_model(returns, vol="Garch", p=1, q=1, dist="t")
        fitted = model.fit(disp="off", show_warning=False)
        cond_vol = fitted.conditional_volatility / 100  # scale back

        result = pd.Series(index=close_prices.index, dtype=float)
        result.loc[cond_vol.index] = cond_vol.values
        return result.ffill().bfill()
    except Exception as e:
        print(f"GARCH fitting failed (using rolling std fallback): {e}")
        # Fallback: simple rolling realized volatility
        returns = close_prices.pct_change()
        return returns.rolling(20).std().bfill()


# ---------------------------------------------------------------------------
# 3. ADR-implied overnight proxy
# ---------------------------------------------------------------------------

def get_adr_implied_return(hk_ticker: str) -> float:
    """
    Returns the overnight return implied by the US-listed ADR for a given
    HK stock, adjusted for the ADR conversion ratio. Acts as an early signal
    for HK open before the local market opens.
    Returns 0.0 (neutral) if the stock has no mapped ADR or fetch fails.
    """
    if hk_ticker not in ADR_PROXIES:
        return 0.0
    try:
        import yfinance as yf
        adr_info = ADR_PROXIES[hk_ticker]
        adr_df = yf.download(adr_info["adr_ticker"], period="5d", progress=False)
        if len(adr_df) < 2:
            return 0.0
        adr_return = adr_df["Close"].pct_change().iloc[-1]
        return float(adr_return)
    except Exception as e:
        print(f"ADR fetch failed for {hk_ticker}: {e}")
        return 0.0


def build_macro_features(close_prices: pd.Series, hk_ticker: str = None) -> dict:
    """Convenience wrapper to compute all macro features at once."""
    flow = get_stock_connect_flow()
    garch_vol = compute_garch_volatility(close_prices)
    adr_return = get_adr_implied_return(hk_ticker) if hk_ticker else 0.0

    return {
        "southbound_net_flow_latest": flow["southbound_net_flow_latest"],
        "southbound_net_flow_5d_avg": flow["southbound_net_flow_5d_avg"],
        "southbound_flow_accelerating": int(flow["southbound_flow_accelerating"]),
        "garch_volatility": garch_vol,
        "adr_implied_return": adr_return,
    }
