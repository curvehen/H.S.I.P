"""
LLM-based commentary generator for HSI and per-stock predictions.
Uses DashScope's OpenAI-compatible endpoint. Fails safe to a neutral
placeholder string on any error — never breaks the prediction pipeline.
"""
from openai import OpenAI
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_TIMEOUT_SECONDS

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    return _client


def _build_prompt(result: dict) -> str:
    stock_lines = "\n".join(
        f"- {s['name']}({s['ticker']}): 訊號={s['signal']}, P(升)={s['p_up']*100:.1f}%, "
        f"強度={s['strength_label']}, 預測回報={s['pred_return_pct']:+.2f}%"
        for s in result.get("_stock_predictions", [])
    )

    return f"""你是一位港股量化分析助理。根據以下結構化預測數據，用繁體中文寫一段
150-200字的分析摘要，涵蓋：(1) HSI大盤方向與可信度，(2) regime狀態意涵，
(3) 值得留意的個股分歧（例如個股方向與大盤不一致的情況）。
不要重複列出原始數字，只做綜合研判，語氣客觀審慎，結尾提醒這只是模型輸出僅供參考。

【HSI大盤】
日期：{result['target_trading_date']}
P(升)：{result['p_up']*100:.1f}%　Regime：{result['regime']}
預測高位：{result['pred_high']:.0f}　預測低位：{result['pred_low']:.0f}
信號強度：{result['signal_strength_label']}（{result['signal_strength_margin']}%）

【個股預測】
{stock_lines}
"""


def generate_market_commentary(result: dict) -> str:
    """Returns LLM-generated commentary, or a safe fallback string on failure."""
    try:
        client = _get_client()
        prompt = _build_prompt(result)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            timeout=LLM_TIMEOUT_SECONDS,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM commentary generation failed: {e}")
        return "（AI分析暫時無法生成，請參考上方結構化數據。）"
