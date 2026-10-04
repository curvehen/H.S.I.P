"""
CCASS shareholding scraper (HKEX SDW).
Best-effort daily snapshot per stock. Wrapped in try/except throughout —
CCASS is the least stable data source and must never break the main pipeline.
Falls back to neutral (0.0) change on any failure.
"""

import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from config import CCASS_DIR

CCASS_URL = "https://www3.hkexnews.hk/sdw/search/searchsdw.aspx"


def fetch_ccass_snapshot(stock_code: str, date: str = None):
    if date is None:
        date = datetime.today().strftime("%Y-%m-%d")

    params = {
        "sc_code": stock_code.zfill(5),
        "sortby": "shareholding",
        "shareholdingdate": date.replace("-", ""),
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        resp = requests.get(CCASS_URL, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table", {"class": "table"})
        if table is None:
            return None

        rows = table.find_all("tr")[1:]
        holdings = []
        for r in rows:
            cols = [c.get_text(strip=True) for c in r.find_all("td")]
            if len(cols) >= 3:
                holdings.append(cols)

        if not holdings:
            return None

        df = pd.DataFrame(holdings)
        shareholding_col = df.columns[-2]
        top10_pct = pd.to_numeric(
            df[shareholding_col].str.replace(",", "").str.replace("%", ""),
            errors="coerce"
        ).head(10).sum()

        result = {"stock_code": stock_code, "date": date, "top10_concentration_pct": float(top10_pct)}
        out_path = CCASS_DIR / f"{stock_code}_{date}.csv"
        pd.DataFrame([result]).to_csv(out_path, index=False)
        return result

    except Exception as e:
        print(f"CCASS fetch failed for {stock_code} on {date}: {e}")
        return None


def get_ccass_change(stock_code: str, days_back: int = 5) -> float:
    files = sorted(CCASS_DIR.glob(f"{stock_code}_*.csv"))
    if len(files) < 2:
        return 0.0
    try:
        recent = pd.read_csv(files[-1])["top10_concentration_pct"].iloc[0]
        older_idx = max(0, len(files) - 1 - days_back)
        older = pd.read_csv(files[older_idx])["top10_concentration_pct"].iloc[0]
        return float(recent - older)
    except Exception:
        return 0.0
