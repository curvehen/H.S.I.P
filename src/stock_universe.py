"""
HSI constituent list + free-float weights.
NOTE: Hang Seng Indexes Company updates constituents/weights quarterly.
Update WEIGHTS_URL check periodically — do not assume this stays static.
"""

# Simplified representative subset with approximate weights (as of last review).
# Production use: re-scrape from https://www.hsi.com.hk index fact sheet monthly.
HSI_CONSTITUENTS = {
    "0700.HK": 0.081,  # Tencent
    "9988.HK": 0.062,  # Alibaba
    "0941.HK": 0.058,  # China Mobile
    "1299.HK": 0.055,  # AIA
    "0388.HK": 0.052,  # HKEX
    "3690.HK": 0.045,  # Meituan
    "0005.HK": 0.044,  # HSBC
    "1810.HK": 0.030,  # Xiaomi
    "2318.HK": 0.028,  # Ping An
    "0016.HK": 0.025,  # SHK Properties
    "0027.HK": 0.020,  # Galaxy Ent
    "0883.HK": 0.020,  # CNOOC
    "1398.HK": 0.019,  # ICBC
    "3988.HK": 0.018,  # Bank of China
    "0002.HK": 0.017,  # CLP
    "0001.HK": 0.016,  # CKH Holdings
}

def get_universe():
    """Returns dict {ticker: weight}. Weights should sum close to 1 for the subset used."""
    total = sum(HSI_CONSTITUENTS.values())
    return {k: v / total for k, v in HSI_CONSTITUENTS.items()}
