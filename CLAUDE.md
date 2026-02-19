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

| 檔案名稱 | 類型 | 用途 | 複雜度 |
|----------|------|------|--------|
| `dpc-starter-train.py` | 訓練 | 訓練基礎模型 | ⭐⭐ |
| `dpc-starter-infer.py` | 推論 | 基礎推論 | ⭐⭐ |
| `byt-ensemble.py` | 推論 | 3 模型融合 | ⭐⭐⭐ |
| `byt-ensemble-59f3b0.py` | 推論 | 2 模型簡單融合 | ⭐⭐ |
| `byt-ensemble-script.py` | 推論 | 極致優化版 | ⭐⭐⭐⭐ |
| `deep-past000.py` | 推論 | 極致優化版 | ⭐⭐⭐⭐ |

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

### 目前進度
- **最新分數**：32.8（公開分數）、排名 903
- **使用模型**：byt5-base + dpc-train-v2-gap.py + dpc-infer-v2-gap.py
- **目前狀態**：Gap 正規化已完成，分數大幅提升

### 分數目標路線

| 階段 | 改進項目 | 實際分數 | 狀態 |
|------|---------|---------|------|
| Baseline | byt5-small, 20 epochs | 23.7 | ✅ 已完成（排名 1451）|
| Step 1 | byt5-base, 10 epochs, cosine LR | 27.7 | ✅ 已完成（排名 1068）|
| Step 2 | Gap 正規化（前處理/後處理）| 32.8 | ✅ 已完成（排名 903）|
| Step 3 | 更多訓練資料 + 更長訓練 | ~35 | ⬜ 下一步 |
| Step 4 | Back-translation + publications 挖掘 | ~37 | ⬜ 待做 |
| Step 5 | Model Soup（多模型融合）| ~39+ | ⬜ 待做 |

### 第一階段：跑通基礎版
- [x] 執行 `starter-train.py`，得到基礎模型
- [x] 用 `starter-infer.py` 產生提交檔
- [x] 提交看看基礎分數 → **23.7**

### 第二階段：改進訓練
- [x] 換成 `byt5-base`（更大模型）→ **27.7**（排名 1068，提升 +4.0）
- [x] 提交 byt5-base 模型，確認分數提升 ✅
- [x] Gap 正規化：統一 `<gap>` / `<big_gap>` 的處理方式 → **32.8**（排名 903，提升 +5.1）
- [ ] 改進句子對齊算法
- [ ] 嘗試增加 epochs 到 20-30（byt5-base）
- [ ] 加入外部資料集訓練（external-dataset/）

### 第三階段：挖掘資料
- [ ] 研究 `publications.csv` 的結構
- [ ] 寫程式提取翻譯對
- [ ] 加入新資料重新訓練
- [ ] 訓練多個模型做融合

### 第四階段：最終優化
- [ ] 用 `byt-ensemble` 融合多個模型
- [ ] 調整後處理規則
- [ ] 最終提交

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

1. **訓練比推論重要 10 倍**：推論程式碼的優化最多提升 2-3%，但好的訓練可以提升 20%+

2. **資料是王道**：挖掘 `publications.csv` 的額外翻譯對是高分選手的秘密武器

3. **模型大小很重要**：`byt5-base` 比 `byt5-small` 大 3 倍，效果明顯更好

4. **Model Soup 有效**：融合多個模型可以穩定提升 3-5%

5. **後處理不可忽視**：正確處理 `<gap>`、分數符號、重複字可以避免扣分
