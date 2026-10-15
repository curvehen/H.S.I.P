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
    CF_API_BASE,
    CF_API_TOKEN,
    CF_MODEL,
    LLM_ANALYSIS_ENABLED,
    LLM_MAX_TOKENS,
    LLM_REQUEST_TIMEOUT_SECONDS,
)

_client = None


def _get_client() -> Optional[OpenAI]:
    """Lazily instantiates the OpenAI-compatible Cloudflare Workers AI
    client. Returns None if the account ID / API token is missing or the
    module is disabled, so callers can fail safely without crashing the
    main prediction pipeline."""
    global _client
    if _client is not None:
        return _client
    if not LLM_ANALYSIS_ENABLED or not CF_API_TOKEN:
        print("LLM_ANALYSIS: CF_ACCOUNT_ID / CF_API_TOKEN not set — commentary disabled.")
        return None
    try:
        _client = OpenAI(
            api_key=CF_API_TOKEN,
            base_url=CF_API_BASE,
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
            model=CF_MODEL,
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
        "另外【請按以下格式輸出】\n\n",
        "明日偏向：上升 / 下跌 / 震盪\n\n",
        "概率：上升__% / 下跌__% / 震盪__%\n\n",
        "信心：高 / 中 / 低\n\n",
        "核心原因：按權重排序，3-5點\n\n",
        "每點註明屬「基本面 / 資金 / 技術 / 情緒 / 外圍」\n\n",
        "關鍵價位：\n\n",
        "支持1 / 支持2：\n\n",
        "阻力1 / 阻力2：\n\n",
        "ATR(14)：\n\n",
        "期指高/低水啟示：\n\n",
        "交易情境：\n\n",
        "A. 造好情境：\n\n",
        "入場條件：\n\n",
        "入場位：\n\n",
        "止蝕位：\n\n",
        "目標位：\n\n",
        "風險回報比：\n\n",
        "B. 造淡情境：\n\n",
        "入場條件：\n\n",
        "入場位：\n\n",
        "止蝕位：\n\n",
        "目標位：\n\n",
        "風險回報比：\n\n",
        "C. 觀望情境：\n\n",
        "咩情況下唔應該入場：\n\n",
        "失效條件：出現咩數據/走勢會推翻上面預測\n\n",
        "明日風險事件：數據、業績、政策、外圍事件\n\n",
        "一句總結：
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
        "而你是專業短線交易員。請分析股票下一交易日走勢。\n\n",
        "請根據技術面、量價、均線、MACD、RSI、KD、籌碼面、消息面、大盤與族群強弱，判斷明日最可能方向。\n\n",
        "請直接輸出：\n\n",
        "1. 明日最可能走勢一句話結論\n\n",
        "2. 上漲 / 震盪 / 下跌機率\n\n",
        "3. 關鍵支撐與壓力位\n\n",
        "4. 看多成立條件\n\n",
        "5. 看空成立條件\n\n",
        "6. 開盤後最重要的 3 個觀察訊號\n\n",
        "7. 若做隔日短線，較合理的交易思路\n\n",
        "8. 本次判斷最不確定的因素\n\n"
        f"數據:{json.dumps(payload, ensure_ascii=False)}"
    )


def generate_market_commentary(result: dict) -> Optional[str]:
    """Generates a Chinese-language HSI market commentary paragraph from
    predict.py's result dict. Returns None on any failure — caller
    (predict.py) treats None as 'no commentary available' and the pipeline
    continues unaffected."""
    system_prompt = (
        "你是專業港股指數交易員。請把恒生指數（HSI）與恒生科技指數（HSTECH）整理成一份高密度晨會筆記。\n\n",
        "請納入：美股隔夜、南向資金、人民幣匯率、權重股動態、重要消息催化、恒指期貨隱含開盤方向。\n\n",
        "輸出格式：\n\n",
        "1. 一句話結論：風險偏好上升/區間整理/風險偏好下降\n\n",
        "2. 3個最重要的驅動因素\n\n",
        "3. 3個最重要的風險\n\n",
        "4. 恒指/恒科指關鍵支撐與阻力\n\n",
        "5. 資金面強弱（南向資金、成交額）\n\n",
        "6. 板塊輪動（誰領漲誰領跌）\n\n",
        "7. 下一交易日最可能情境與機率\n\n",
        "8. 多方成立條件\n\n",
        "9. 空方成立條件\n\n",
        "10. 開盤後最值得盯的3個訊號\n\n",
        "11. 交易策略建議（偏多/偏空/觀望）\n\n",
        "12. 判斷可信度\n\n",
        "規則：\n\n",
        "- 只根據可取得的事實，不要虛構即時數據\n\n",
        "- 事實與推論分開\n\n",
        "- 內容短、狠、準，不要像研究報告\n\n",
        "- 若資料不足，先列出缺口，再做保守判斷"
    )
    user_prompt = _build_market_prompt(result)
    return _safe_chat_completion(system_prompt, user_prompt)


def generate_stock_summary(stock: dict) -> Optional[str]:
    """Generates a one-line Chinese summary for a single stock prediction.
    Applies identically to HSI constituents and watchlist tickers. Returns
    None on failure."""
    system_prompt = (
        "你是專業港股短線交易員。請把【股票名稱 / 股票代號】整理成一份高密度港股 stock summary，格式像交易台晨會筆記，重點不是介紹公司，而是幫我快速判斷：這檔股票現在是否值得交易、交易邏輯是什麼、風險在哪裡。"
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
