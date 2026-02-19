# Deep Past Challenge - 古亞述語機器翻譯

Kaggle 比賽：將 4000 年前的**古亞述語（Old Assyrian）**轉寫翻譯成**英文**。

比賽連結：[Deep Past Initiative - Machine Translation](https://kaggle.com/competitions/deep-past-initiative-machine-translation)

---

## 分數進度

| 版本 | 日期 | 分數 | 排名 | 提升 | 做法 |
|------|------|------|------|------|------|
| v0 - Baseline | 2026-02-19 | 23.7 | 1451 | — | byt5-small, 20 epochs, 基礎 starter 腳本 |
| v1 - 換大模型 | 2026-02-19 | 27.7 | 1068 | +4.0 | byt5-base, 10 epochs, cosine LR schedule |
| v2 - Gap 正規化 | 2026-02-19 | 32.8 | 903 | +5.1 | Gap 正規化 + 改進前後處理 |

**評分公式**：`score = sqrt(BLEU × chrF++)`

---

## 各版本詳細做法

### v0 - Baseline（23.7）

直接使用比賽提供的 starter code，沒有任何修改。

- **模型**：`google/byt5-small`（~300M 參數）
- **訓練**：20 epochs, batch size 4, LR 2e-4
- **推論**：beam search = 4, 基礎後處理
- **腳本**：`dpc-starter-train.py` + `dpc-starter-infer.py`

### v1 - 換大模型（27.7, +4.0）

最簡單的改進：把模型從 small 換成 base。

- **模型**：`google/byt5-base`（~580M 參數，比 small 大 2 倍）
- **訓練**：10 epochs, batch size 8, cosine LR schedule with warmup
- **句子對齊**：加入 sentence-transformers 做語意比對，從 1,561 份文件展開成 ~8,300 句子對
- **腳本**：`dpc-train-v2-gap.py`（早期版本）

### v2 - Gap 正規化（32.8, +5.1）

針對破損文本（gap）做全面的正規化處理，統一訓練和推論的格式。

- **前處理改進**：
  - 統一各種破損標記：`[...]`, `…`, `xx`, `x` → `<gap>` / `<big_gap>`
  - 訓練資料中 1,652 筆轉寫、2,098 筆翻譯被正規化
- **後處理改進**：
  - 特殊字元正規化（ḫ→h, 下標數字轉一般數字）
  - Gap token 還原為標準格式
  - 移除不必要的註解（fem, plur, ?, !）
  - 分數轉換（0.5→½, 0.25→¼, 0.75→¾）
  - 重複詞移除、標點符號清理
- **腳本**：`dpc-train-v2-gap.py` + `dpc-infer-v2-gap.py`

---

## 專案結構

```
deep_past_challenge/
├── dpc-train-v2-gap.py          # 目前使用的訓練腳本
├── dpc-infer-v2-gap.py          # 目前使用的推論腳本
├── dpc-train-v2-gap.log         # 訓練日誌
├── CLAUDE.md                    # 完整比賽筆記與策略
├── README.md                    # 本檔案
│
├── deep-past-initiative-machine-translation/   # 比賽資料集
│   ├── train.csv                # ~1,500 份有翻譯的文件
│   ├── test.csv                 # ~4,000 個待翻譯句子
│   ├── published_texts.csv      # ~8,000 份轉寫（無翻譯）
│   └── publications.csv         # ~900 本學術論文 OCR（554 MB）
│
├── external-dataset/            # 外部補充資料
│   ├── akkadian_corpus.csv      # 擴充語料（6.6 MB）
│   ├── akkadian_dictionary.csv  # 字典（3.9 MB）
│   └── output_gap_big_gap5.csv  # Gap 處理資料
│
└── reference/                   # 參考用推論腳本
    ├── byt-ensemble-script.py   # 3 模型融合
    ├── deep-past000.py          # 極致優化推論
    └── ...
```

---

## 下一步計畫

- [ ] 增加訓練 epochs（20~30）
- [ ] 整合 external-dataset 的額外訓練資料
- [ ] 多模型融合（Model Soup）
- [ ] 挖掘 publications.csv 提取翻譯對
- [ ] Back-translation（英→古亞述語反向翻譯）

---

## 技術棧

- **模型**：Google ByT5（byte-level T5）
- **框架**：HuggingFace Transformers + Trainer
- **訓練環境**：Kaggle（Tesla P100 GPU）
- **語言**：Python
