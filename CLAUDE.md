# Deep Past Challenge 比賽筆記

## 零、環境設定

- 執行 Python 程式時，使用 `python`，不要使用 `python3`

## 一、比賽簡介

### 這是什麼比賽？
- **目標**：訓練 AI 把 4000 年前的**古亞述語**翻譯成**英文**
- **獎金**：總計 $50,000 美金（第一名 $15,000）
- **截止日期**：2026/3/23

### 為什麼重要？
- 全世界只有不到 12 個專家看得懂古亞述語
- 博物館裡有上萬份泥板未被翻譯
- 你的 AI 可以幫助解讀人類歷史

---

## 二、資料集結構

### 主要檔案

| 檔案 | 內容 | 用途 |
|------|------|------|
| `train.csv` | ~1,500 份文件（有翻譯）| 訓練模型 |
| `test.csv` | ~4,000 個句子（無翻譯）| 產生提交 |
| `published_texts.csv` | ~8,000 份轉寫（無翻譯）| 額外資料 |
| `publications.csv` | ~900 本學術論文的 OCR | 挖掘翻譯對 |

### 資料格式

**輸入（古亞述語轉寫）**：
```
um-ma A-šur-GAL / a-na Ku-zi-a / qí-bi-ma
```

**輸出（英文翻譯）**：
```
Thus says Ashur-GAL to Kuzia: speak!
```

### 特殊符號對照表

| 符號 | 意思 | 例子 |
|------|------|------|
| `{d}` | 神的名字 | `{d}UTU` = 太陽神 |
| `{m}` | 男性人名 | `{m}A-šur` = 亞述先生 |
| `{ki}` | 地名 | `A-lim{ki}` = 阿林姆城 |
| `[ ]` | 破損部分 | `[x x]` = 看不清楚 |
| `<gap>` | 小破損 | 缺一個字 |
| `<big_gap>` | 大破損 | 缺很多字 |

---

## 三、範例程式碼總覽

### 檔案清單

| 檔案名稱 | 類型 | 用途 | 狀態 |
|----------|------|------|------|
| `dpc-starter-train.py` | 訓練 | 基礎模型（byt5-small）| baseline, 24.1 分 |
| `dpc-starter-infer.py` | 推論 | 基礎推論 | baseline |
| `dpc-train-v2-gap.py` | 訓練 | byt5-base + gap 正規化 | **31.8 分（目前最佳）** |
| `dpc-infer-v2-gap.py` | 推論 | gap 前/後處理 | 搭配 v2 使用 |
| `dpc-train-v3-oare.py` | 訓練 | v2 + OARE 外部資料 | 失敗（domain mismatch） |
| `dpc-train-v4-tuned.py` | 訓練 | v2 + 參數優化（LS=0.1） | 待跑 |
| `dpc-train-v4b-ls02.py` | 訓練 | v4 變體（LS=0.2） | 待跑 |
| `dpc-infer-v4-improved.py` | 推論 | beam=8 + MBR decoding | 待跑 |
| `reference/` | 參考 | 高分選手腳本（不直接使用）| — |

### 重要發現

> **所有推論程式都是用「別人訓練好的模型」，真正決定分數的是「模型怎麼訓練」！**

---

## 四、訓練程式分析 (`dpc-starter-train.py`)

### 基礎流程

```python
# 1. 讀取資料
train_df = pd.read_csv("train.csv")

# 2. 句子對齊（把文章切成句子）
train_expanded = simple_sentence_aligner(train_df)

# 3. 載入預訓練模型
model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")

# 4. 訓練
trainer.train()

# 5. 儲存模型
trainer.save_model("./byt5-akkadian-model")
```

### 關鍵參數

| 參數 | 預設值 | 意思 |
|------|--------|------|
| `MODEL_NAME` | google/byt5-small | 使用的預訓練模型 |
| `MAX_LENGTH` | 512 | 最長處理 512 個字元 |
| `BATCH_SIZE` | 4 | 一次處理 4 筆資料 |
| `EPOCHS` | 20 | 訓練 20 輪 |
| `LEARNING_RATE` | 2e-4 | 學習速度 |

### starter-train 的不足

| 項目 | 目前做法 | 可改進方向 |
|------|----------|-----------|
| 模型大小 | byt5-small | 換成 byt5-base 或 large |
| 訓練資料 | 只用 train.csv | 挖掘 publications.csv |
| 訓練輪數 | 20 epochs | 40-50 epochs |
| 句子對齊 | 簡單規則 | 用 Sentence-BERT |
| 學習率 | 固定值 | Warmup + Cosine Decay |

---

## 五、推論程式比較

### 技術特性對照表

| 特性 | starter-infer | byt-ensemble | 59f3b0 | script/000 |
|------|---------------|--------------|--------|------------|
| 模型數量 | 1 | 3 | 2 | 1 |
| Model Soup | ❌ | ✅ | ✅ | ❌ |
| Beam Search | 4 | 8 | 4 | 8（自適應）|
| 前處理 | ❌ | ✅ | ✅ | ✅ 向量化 |
| 後處理 | ✅ 基礎 | ✅ 詳細 | ❌ | ✅ 極致 |
| Mixed Precision | ❌ | ❌ | ❌ | ✅ |
| BetterTransformer | ❌ | ❌ | ❌ | ✅ |
| Bucket Batching | ❌ | ❌ | ❌ | ✅ |

### Model Soup（模型融合）解釋

```python
# 把多個模型的參數「加權平均」
weights = [0.42, 0.41, 0.17]  # 三個模型的權重

for key in model_params:
    new_param = (weights[0] * model1[key] +
                 weights[1] * model2[key] +
                 weights[2] * model3[key])
```

**白話**：像是把三個學霸的答案按成績比例混合，正確率更高。

---

## 六、分數提升策略

### 各策略的重要性

```
總分數 = 模型品質(80%) + 推論技巧(15%) + 後處理(5%)
              ↑
           最重要！
```

### 改進方向與預估效果

| 改進方向 | 預估分數提升 | 難度 |
|----------|-------------|------|
| 換更大的模型 | +5~10% | ⭐ |
| 訓練更久 | +2~5% | ⭐ |
| 挖掘額外資料 | +10~20% | ⭐⭐⭐⭐ |
| 更好的句子對齊 | +5~10% | ⭐⭐⭐ |
| 模型融合 | +3~5% | ⭐⭐ |
| 推論優化 | +1~2% | ⭐⭐ |
| 後處理 | +0.5~1% | ⭐ |

---

## 七、訓練改進建議

### Level 1：簡單改進

```python
# 1. 換更大的模型
MODEL_NAME = "google/byt5-base"  # 原本是 small

# 2. 訓練更久
EPOCHS = 40  # 原本是 20

# 3. 用更好的學習率策略
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps
)
```

### Level 2：更好的句子對齊

```python
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment

model = SentenceTransformer('all-MiniLM-L6-v2')

def smart_sentence_align(src_lines, tgt_sents):
    # 計算語意相似度
    src_emb = model.encode(src_lines)
    tgt_emb = model.encode(tgt_sents)
    similarity = cosine_similarity(src_emb, tgt_emb)

    # 用匈牙利算法找最佳配對
    row_ind, col_ind = linear_sum_assignment(-similarity)
    return [(src_lines[i], tgt_sents[j]) for i, j in zip(row_ind, col_ind)]
```

### Level 3：挖掘額外資料

```python
# 從 publications.csv 提取翻譯（554 MB！）
pub = pd.read_csv("publications.csv")
akkadian_pages = pub[pub['has_akkadian'] == True]

# 用正則表達式或 LLM 提取翻譯對
# 這是最難但收益最大的部分！
```

### Level 4：多模型訓練與融合

```python
# 訓練多個不同的模型
train_model_a(all_data, seed=42)      # 模型 A
train_model_b(all_data, seed=123)     # 模型 B（不同種子）
train_model_c(gap_data, seed=42)      # 模型 C（專攻破損文本）

# 用 Model Soup 融合
```

---

## 八、參賽路線圖與目標

### 最終結果（比賽已結束）
- **最終 Private Score**：**32.4618**、最終排名 **1356**（共 8 次提交）
- **最佳模型**：byt5-base + dpc-train-v2-gap.py + dpc-infer-v2-gap.py（v2-gap）
- **選擇的提交**：v2-gap（public 與 private 皆最高，正確避開了 v3/v4 的假指標陷阱）
- **截止日期**：2026/3/23（已結束）
- **GPU 預算**：~83 小時

### 分數歷程（Kaggle 真實 public / private）

| 版本 | 改進項目 | Public | Private | 狀態 |
|------|---------|--------|---------|------|
| Baseline | byt5-small, 20 epochs | 24.1 | 23.9 | ✅ |
| v1 | byt5-base, 10 epochs, cosine LR | 28.2 | 27.4 | ✅ |
| v2-gap | + gap 正規化（前/後處理）| **31.9** | **32.5** | ✅ 最佳（最終提交）|
| v3-oare | + OARE 外部資料 | 31.4 | 31.6 | ❌ 未超越 v2（domain mismatch）|
| v4-tuned | + LS=0.1, warmup, 15ep | 30.6 | 31.3 | ❌ 未超越 v2（eval chrF=45.5 但降分）|
| v4b-ls02 | + LS=0.2 變體 | — | — | ⬜ 未跑（v4 已證明方向無效）|
| v5-bidir | + 雙向訓練 | — | — | ⬜ 未進行 |
| v6-final | + continue training / soup | — | — | ⬜ 未進行 |

### v3-oare 失敗教訓
- OARE 資料與 train.csv domain 不同，加入後模型偏離測試集分佈
- eval chrF 持續上升但公開分數未提升、checkpoint 間波動大（public 27.7~31.4，未超越 v2 的 31.9）
- **結論**：外部資料需謹慎篩選，回到 train.csv only 路線改用參數優化

### v4-tuned 訓練結果（已提交，未超越 v2）
- **GPU**：P100，25399.7s（~7hr），15 epochs / 12285 steps
- **eval chrF 歷程**：36.4 → 43.1 → 45.3 → 45.1 → **45.5**（全專案最高）
- **提交結果**：public **30.6** / private **31.3**，**雙雙低於 v2**（31.9 / 32.5）
- **診斷**：離線指標創新高但 leaderboard 反降——label smoothing 可能讓輸出過於保守；
  與 v3 同屬「離線指標 ≠ 排行榜」的教訓
- **結論**：v4 方向無效，v4b-ls02 不再嘗試；確定 v2-gap 為最終最佳模型

### Phase 1：參數優化（v4）— 結果：失敗（未超越 v2）
- [x] 建立 `dpc-train-v4-tuned.py`（LS=0.1, warmup=200, 15ep, eval 優化）
- [x] 建立 `dpc-train-v4b-ls02.py`（LS=0.2 變體）
- [x] 建立 `dpc-infer-v4-improved.py`（beam=8, length_penalty=1.3, MBR）
- [x] 在 Kaggle 跑 v4-tuned（~7hr GPU，完成）
- [x] 用 v4-infer 推論 v4-tuned 並提交 → public 30.6（低於 v2，方向作廢）
- [x] 決定不跑 v4b-ls02（v4 已證明 LS 方向無效）

### v4 vs v2 參數對比

| 參數 | v2 | v4 | 理由 |
|------|-----|-----|------|
| `label_smoothing_factor` | 0 | **0.1** (v4b: 0.2) | 防過擬合 |
| `warmup_steps` | 0 | **200** | 穩定初始訓練 |
| `EPOCHS` | 10 | **15** | 防欠擬合 |
| `eval_strategy` | epoch | **steps（每 3ep）** | 省時間 |
| eval set | 10%（~700） | **300 固定** | 省時間 |
| `generation_num_beams`（eval） | 4 | **1（greedy）** | 省時間 |
| `generation_max_length` | 未設 | **512** | ByT5 必須 |
| StderrLogCallback | 無 | **有** | Kaggle Logs 可見 |

### v4-infer vs v2-infer 參數對比

| 參數 | v2 | v4 |
|------|-----|-----|
| `num_beams` | 4 | **8** |
| `length_penalty` | 1.0 | **1.3** |
| `repetition_penalty` | 1.0 | **1.2** |
| MBR decoding | 無 | **有（可選）** |

### Phase 2：雙向訓練 + Model Soup（v5）— 目標 37-38
- [ ] 建立 `dpc-train-v5-bidir.py`（雙向資料 + adafactor）
- [ ] Model Soup（v4a + v4b + v5 加權平均）
- [ ] 推論並提交

### Phase 3：進階訓練（v6）— 目標 39+
- [ ] Continue training（低 LR 繼續訓練最佳模型）
- [ ] Back-translation（用最佳模型翻譯 published_texts）
- [ ] 最終 Model Soup + MBR

### Phase 4：最終提交（第 20-23 天）
- [ ] 最終 soup + MBR + 最佳推論參數
- [ ] 確保有 2-3 個安全提交

---

## 九、評分方式

### 公式
```
分數 = sqrt(BLEU × chrF++)
```

- **BLEU**：詞級準確度（翻譯的「詞」對不對）
- **chrF++**：字元級準確度（翻譯的「字母」對不對）
- **幾何平均**：兩個分數相乘後開根號

### 提交格式
```csv
id,translation
0,Thus Kanesh, say to the...
1,In the letter of the City...
```

---

## 十、重要連結

- 比賽頁面：[Kaggle Deep Past Challenge](https://kaggle.com/competitions/deep-past-initiative-machine-translation)
- 評分實作：[BLEU 和 chrF++ 的幾何平均值](https://www.kaggle.com/code/ryanholbrook/geometric-mean-of-bleu-and-chrf)
- 深入了解：[Deep Past Initiative 官網](https://www.deeppastinitiative.org/)

---

## 十一、關鍵結論

1. **離線指標 ≠ 排行榜**（本次最大教訓）：v3 與 v4 的 eval chrF 都上升，public/private 卻下降。必須以 leaderboard 驗證假設，不能只看 dev set。

2. **資料品質 > 資料數量**：v3-oare 證明亂加外部資料會 domain mismatch，反而降分

3. **模型大小很重要**：`byt5-base` 比 `byt5-small` 大 3 倍，效果明顯更好（v1 一口氣 +4.0）

4. **早期改進 CP 值最高**：換大模型（+4.0）與 gap 正規化（+3.7）帶來最大躍進；越後期邊際效益越低，甚至為負。

5. **後處理不可忽視**：正確處理 `<gap>`、分數符號、重複字可以避免扣分（v2-gap 的關鍵）

6. **誠實面對失敗**：v2-gap 始終是最佳模型；最終正確選擇它提交（private 32.46）。Model Soup、雙向訓練、back-translation 等 Phase 2-4 計畫因比賽結束未驗證，列為未來方向而非已證實結論。
