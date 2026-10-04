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
