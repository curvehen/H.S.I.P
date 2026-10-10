"""
LLM Analysis — generates concise Chinese-language market commentary and
per-stock summaries using a DashScope-compatible OpenAI-style endpoint
(Qwen models via Alibaba Cloud's international DashScope gateway).

Design principles:
  - Fail-safe by default: any exception (missing API key, network error,
    timeout, malformed response) returns None rather than raising, so a
    commentary failure NEVER breaks predict.py's main pipeline.
  - Only called on "official" runs (see predict.py's LLM_ENABLED_RUN_MODES)
    to control API cost and guarantee reproducibility within a trading day.
  - Prompts are built entirely from already-computed prediction dict fields
    (no re-fetching of data), keeping the LLM a pure "summarizer" layer.
  - Covers BOTH the HSI-level market commentary AND independent per-stock
    summaries for every tracked ticker — HSI constituents AND watchlist
    tickers (e.g. 1211.HK, 0968.HK) alike. Watchlist-vs-constituent status is
    carried through via the "is_constituent" field already present in the
    stock dict produced by predict.py; this module makes no distinction in
    how it treats the two groups (isolation from HSI bottom-up weighting is
    enforced upstream in predict.py / stock_universe.py, not here).

FIX LOG (this revision):
  - Corrected `config` import names: the previous version imported
    LLM_MODEL / LLM_BASE_URL / LLM_TIMEOUT_SECONDS, which do not exist in
    config.py and raised ImportError on every run, silently disabling LLM
    commentary end-to-end. Now imports the actual names defined in
    config.py: DASHSCOPE_MODEL_NAME, DASHSCOPE_API_BASE,
    LLM_REQUEST_TIMEOUT_SECONDS, LLM_MAX_TOKENS, DASHSCOPE_API_KEY,
    LLM_ANALYSIS_ENABLED.
  - Reused config.py's DASHSCOPE_API_KEY / LLM_ANALYSIS_ENABLED instead of
    re-reading the environment variable independently.
  - generate_stock_summaries_batch() no longer hard-caps at 5 stocks by
    default; pass max_stocks=None (default) to summarize every stock given
    to it (constituents + watchlist). Pass an explicit integer to cap API
    usage if needed.
"""

import json
from typing import Optional

from openai import OpenAI

from config import (
    DASHSCOPE_API_BASE,
    DASHSCOPE_API_KEY,
    DASHSCOPE_MODEL_NAME,
    LLM_ANALYSIS_ENABLED,
    LLM_MAX_TOKENS,
    LLM_REQUEST_TIMEOUT_SECONDS,
)

_client = None


def _get_client() -> Optional[OpenAI]:
    """Lazily instantiates the OpenAI-compatible DashScope client. Returns
    None if the API key is missing or the module is disabled, so callers can
    fail safely without crashing the main prediction pipeline."""
    global _client
    if _client is not None:
        return _client
    if not LLM_ANALYSIS_ENABLED or not DASHSCOPE_API_KEY:
        print("LLM_ANALYSIS: DASHSCOPE_API_KEY not set — commentary disabled.")
        return None
    try:
        _client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_API_BASE,
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        )
        return _client
    except Exception as e:
        print(f"LLM_ANALYSIS: client init failed: {e}")
        return None


def _safe_chat_completion(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Single shared call wrapper with full fail-safe handling. Any failure
    (timeout, network error, quota exceeded, malformed response) returns
    None instead of raising, per the project's fail-safe design contract."""
    client = _get_client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=DASHSCOPE_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=LLM_MAX_TOKENS,
            temperature=0.4,
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            print("LLM_ANALYSIS: empty response content.")
            return None
        return content.strip()
    except Exception as e:
        print(f"LLM_ANALYSIS: chat completion failed: {e}")
        return None


def _build_market_prompt(result: dict) -> str:
    """Builds a compact, numeric-grounded prompt from predict.py's HSI-level
    result dict. Only fields already computed upstream are referenced — the
    LLM performs no independent calculation, only narrative synthesis.
    Uses .get() throughout so missing/renamed keys degrade to None rather
    than raising KeyError."""
    payload = {
        "交易日期": result.get("target_trading_date"),
        "市場Regime": result.get("regime"),
        "校準後訊號": result.get("calibrated_signal"),
        "P_上升": result.get("p_up"),
        "訊號強度": result.get("signal_strength_label"),
        "模型信心分數": result.get("signal_confidence"),
        "建議倉位百分比": result.get("position_size_pct"),
        "現價": result.get("last_close"),
        "預測低位": result.get("pred_low"),
        "預測高位": result.get("pred_high"),
        "預測收市": result.get("pred_close"),
        "預測波幅百分比": result.get("pred_range_pct"),
        "是否值博": result.get("worth_trading"),
        "值博原因": result.get("worth_trading_reason"),
        "歷史命中率": result.get("hit_rate"),
        "Bottom_up覆蓋權重": result.get("bottom_up_coverage_weight"),
        "數據是否非即時": result.get("is_stale"),
    }
    return (
        "以下係一個量化模型對恆生指數下一個交易日嘅預測數據(JSON 格式)。"
        "請用繁體中文、以專業但淺白嘅語氣,寫一段 100-150 字嘅市場分析摘要,"
        "需要包含:市場氛圍判斷、主要支持嗰個訊號嘅數據點、以及一句風險提示。"
        "不要重複列出原始數字(讀者已經睇到表格),專注於解讀同埋脈絡化嗰啲數字。"
        "加任何免責聲明以外嘅建議用語(例如「建議買入」),只做客觀分析。\n\n"
        f"數據:{json.dumps(payload, ensure_ascii=False)}"
    )


def _build_stock_prompt(stock: dict) -> str:
    """Builds a short per-stock summary prompt from predict_single_stock() /
    predict_watchlist()'s output dict. The same prompt template is used for
    both HSI constituents and watchlist tickers; "是否HSI成份股" flags which
    group a given ticker belongs to so the LLM can phrase accordingly."""
    payload = {
        "代號": stock.get("ticker"),
        "名稱": stock.get("name"),
        "行業": stock.get("sector"),
        "方向": stock.get("signal"),
        "P_上升": stock.get("p_up"),
        "訊號強度": stock.get("strength_label"),
        "現價": stock.get("last_close"),
        "預測收市": stock.get("pred_close"),
        "預測變動百分比": stock.get("pred_return_pct"),
        "RSI": stock.get("rsi"),
        "歷史命中率": stock.get("hit_rate"),
        "是否HSI成份股": stock.get("is_constituent"),
    }
    return (
        "以下係一隻港股嘅模型預測數據(JSON 格式)。"
        "請用繁體中文寫一句 30-50 字嘅精簡摘要,講出方向同關鍵理據(例如 RSI 水平、訊號強度),"
        "加投資建議字眼。\n\n"
        f"數據:{json.dumps(payload, ensure_ascii=False)}"
    )


def generate_market_commentary(result: dict) -> Optional[str]:
    """Generates a Chinese-language HSI market commentary paragraph from
    predict.py's result dict. Returns None on any failure — caller
    (predict.py) treats None as 'no commentary available' and the pipeline
    continues unaffected."""
    system_prompt = (
        "你係一位資深港股量化分析師,負責將模型輸出嘅數據轉化為簡潔、客觀嘅市場評論。"
        "你唔提供投資建議,只係解讀模型數據同埋市場脈絡。"
    )
    user_prompt = _build_market_prompt(result)
    return _safe_chat_completion(system_prompt, user_prompt)


def generate_stock_summary(stock: dict) -> Optional[str]:
    """Generates a one-line Chinese summary for a single stock prediction.
    Applies identically to HSI constituents and watchlist tickers. Returns
    None on failure."""
    system_prompt = (
        "你係一位港股分析助手,負責將單一股票嘅模型預測數據濃縮成一句精簡評論。"
    )
    user_prompt = _build_stock_prompt(stock)
    return _safe_chat_completion(system_prompt, user_prompt)


def generate_stock_summaries_batch(stocks: list, max_stocks: Optional[int] = None) -> dict:
    """Generates independent LLM summaries for stock predictions — covers
    HSI constituents AND watchlist tickers (e.g. 1211.HK, 0968.HK) equally,
    since both groups' prediction dicts share the same schema consumed by
    _build_stock_prompt().

    Args:
        stocks: list of per-stock prediction dicts (constituents + watchlist
            combined, as produced by predict.py's run_daily_prediction()).
        max_stocks: optional cap on how many stocks to summarize, ranked by
            absolute predicted return, to control API cost/latency/quota on
            days with many tracked tickers. None (default) summarizes every
            stock passed in.

    Returns:
        {ticker: summary_or_None}. A None value means that ticker's LLM call
        failed or timed out — this is NOT raised as an error, per the
        project's fail-safe contract; callers should render it as "N/A" or
        omit the LLM line for that ticker.
    """
    if not stocks:
        return {}
    sorted_stocks = sorted(stocks, key=lambda s: abs(s.get("pred_return_pct", 0) or 0), reverse=True)
    selected = sorted_stocks if max_stocks is None else sorted_stocks[:max_stocks]
    summaries = {}
    for s in selected:
        ticker = s.get("ticker")
        if not ticker:
            continue
        summaries[ticker] = generate_stock_summary(s)
    return summaries
