# HSI 全方位次日預測系統

架構：Colab負責訓練（手動觸發）→ GitHub Actions每日自動inference + drift監察。

## 涵蓋數據維度
- 技術指標 + K線形態 (TA-Lib)
- 跨市場因子：美股期指(ES Futures)、VIX
- 南北水資金流 (akshare, East Money數據源)
- CCASS個股籌碼集中度變化 (HKEX SDW)
- 新聞情緒 (FinBERT)
- GARCH(1,1)條件波動率
- ADR隔夜proxy (阿里/京東等雙重上市股)
- 個股Bottom-up彙整 (按HSI權重加權)

## 輸出
- 次日入場位、高位、低位、收市位
- 值博與否判斷 (基於預測幅度 + Meta-confidence)
- Risk:Reward比例

## 設定步驟
1. GitHub建立Public repo，加入全部`src/`檔案
2. 生成Fine-grained PAT (Contents: Read and write)
3. Repo → Settings → Actions → Workflow permissions → Read and write permissions
4. Colab：clone repo，設定TA-Lib，`pip install -r requirements.txt`
5. 首次訓練：`cd src && python train_model.py`，然後push `models/` 返repo
6. GitHub Actions會自動用已訓練模型做每日predict

## 已知限制 (務必了解)
- CCASS爬蟲依賴HKEX網頁結構，隨時可能因網站改版失效 — 已設計fail-safe回退至中性值(0.0)
- akshare南北水數據源非官方，結構可能不定期變動
- FinBERT只處理英文新聞，中文財經新聞覆蓋不足
- 成分股只涵蓋16隻代表性重磅股，非全部約80隻 — 可自行擴充`config.py`的`HSI_CONSTITUENTS`
- **次日單一交易日預測屬高噪音問題，不應假設此系統能達到高準繩度；建議作為輔助參考，並持續用`evaluate_drift.py`監察實際表現**




情況 A：加 Watchlist 股票（No HSI weight impact）
Step	檔案	動作	執行環境
1	stock_universe.py	在 WATCHLIST_INFO 加入新 ticker："XXXX.HK": {"sector": "...", "name": "..."}	本機/編輯器，git commit
2	config.py	可選：若新股票在美股有 ADR，於 ADR_PROXIES 加入 "XXXX.HK": {"adr_ticker": "YYYY", "ratio": N}；否則跳過	本機/編輯器
3	train_model.py	直接執行（get_watchlist() 會自動把新 ticker 含進 all_tickers）	Google Colab（依其 docstring 強制要求）
4	Colab 訓練完後	手動下載/push 新產生的 models/stocks/XXXX_HK_* 模型檔回 repo	Google Colab → 手動 push 或 Colab 內 git push
5	run_daily_backtest.py	執行一次（本機或下次 GitHub Actions official run 自動跑），讓 get_watchlist() 迴圈把新 ticker 的歷史回測寫進 backtest_stocks.csv	GitHub Actions（自動，下次 official run）或本機測試
6	hit_rate_tracker.py	自動在同一個 official run 中接著執行，累積 ≥5 個樣本後命中率才會顯示（非 N/A）	GitHub Actions（自動）
7	驗證	檢查下次 email（email_report.py）的「獨立觀察名單」區塊是否出現新股票	無需操作，自動
不用改： predict.py（predict_watchlist() 已自動迭代 get_watchlist()）、email_report.py（已通用渲染）、daily_predict.yml（流程已固定）。

情況 B：加 HSI 成分股（HSI weight impact）
Step	檔案	動作	執行環境
1	stock_universe.py	在 HSI_CONSTITUENTS_INFO 加入 "XXXX.HK": {"weight": 0.0XX, "sector": "...", "name": "..."}，權重需來自官方 HSI 指數公告（本工具無法提供）	本機/編輯器
2	config.py	可選：若新股票有 ADR，於 ADR_PROXIES 加入對應項	本機/編輯器
3	train_model.py	執行（get_universe() 自動含新 ticker，get_universe() 內部會重新正規化權重，不用手動調整其他股票權重）	Google Colab（強制）
4	Colab 訓練完後	下載/push 新模型檔回 repo	Google Colab → push
5	run_daily_backtest.py	執行一次，讓新成分股進入 backtest_stocks.csv（走 for ticker in universe: 迴圈）	GitHub Actions（自動）或本機測試
6	hit_rate_tracker.py	同情況 A，累積樣本後顯示命中率	GitHub Actions（自動）
7	predict.py	無需改碼，predict_bottom_up() 自動把新 ticker 計入加權 bottom-up 指數預測	無需操作
8	驗證	檢查下次 email 的 HSI 成分股表格、bottom-up 加權計算是否正確反映新權重分配	無需操作
不用改： predict.py、email_report.py、daily_predict.yml、signal_generator.py（不涉及個股清單）。
