# %% [markdown]
# # Deep Past Initiative – 機器翻譯（訓練筆記本）
# 
# 本筆記本是此 Kaggle 競賽的 **入門 / 基準線** 方案。
# 
# 主要想法：
# - 使用 **ByT5** 在字元（位元組）層級處理含雜訊的阿卡德語轉寫
# - 執行 **簡易句子對齊** 以增加訓練資料
# - 使用 HuggingFace `Trainer` 進行微調
# 
# 
# 推論程式碼在[這裡](https://www.kaggle.com/code/takamichitoda/dpc-starter-infer)。

# %%
!pip install evaluate sacrebleu

# %%
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")

# %%
import os
import gc
import re
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from sentence_transformers import SentenceTransformer, util
import evaluate

# %%
class Config:
    # byt5-base: 參數量約 580M，比 small (300M) 大 ~2 倍，翻譯品質明顯提升
    MODEL_NAME = "google/byt5-base"

    MAX_LENGTH = 512

    BATCH_SIZE = 8       # effective batch size（透過 gradient accumulation 達成）
    EPOCHS = 10           # base 模型更大，10 epochs 足夠收斂
    LEARNING_RATE = 2e-4
    OUTPUT_DIR = "./byt5-base-akkadian"

# %%
# 固定隨機種子（確保可重現性）。
def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
seed_everything()

# %%
INPUT_DIR = "/kaggle/input/competitions/deep-past-initiative-machine-translation" if os.path.exists("/kaggle/input/competitions/deep-past-initiative-machine-translation") else "./deep-past-initiative-machine-translation"
train_df = pd.read_csv(f"{INPUT_DIR}/train.csv")
test_df = pd.read_csv(f"{INPUT_DIR}/test.csv")

# %%
print(f"Original Train Data: {len(train_df)} docs")

# %%
def normalize_gaps(text):
    """
    統一各種破損/缺失標記為 <gap> 和 <big_gap>。
    訓練資料中有 [...], …, xx, x 等不同表示方式，全部正規化。
    """
    if pd.isna(text):
        return ""
    text = str(text)
    # 大缺損（[...] 要在 ... 之前處理，否則會變成 [<big_gap>]）
    text = re.sub(r'\[\.\.\.\]', '<big_gap>', text)      # [...] 先處理
    text = re.sub(r'\.{3,}', '<big_gap>', text)          # ... or .... etc
    text = re.sub(r'…+', '<big_gap>', text)              # … or ……
    # 小缺損：xx, 獨立的 x
    text = re.sub(r'xx+', '<gap>', text)                 # xx, xxx, ...
    text = re.sub(r'(?<=\s)x(?=\s)', '<gap>', text)      # isolated x between spaces
    # 合併相鄰 gap
    text = re.sub(r'<gap>\s*<gap>', '<big_gap>', text)
    text = re.sub(r'<big_gap>\s*<big_gap>', '<big_gap>', text)
    # 清理多餘空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def proportional_sentence_aligner(df):
    """
    【策略核心】
    將訓練資料的「文件」拆分成「句子對」，讓訓練格式更接近測試集（單句翻譯）。

    方法：
    1. 先用標點符號拆分英文翻譯成多個句子
    2. 如果 Akkadian 有換行符號且數量匹配 → 直接配對（原始方法）
    3. 否則，按英文句子的字元長度比例，等比切割 Akkadian 文字
       例：英文 3 句，比例 30%/50%/20% → Akkadian 也按 30%/50%/20% 切
    4. 切割時盡量在空格處切，避免切斷單字
    """
    aligned_data = []
    exact_split = 0       # 換行符號完美匹配
    proportional_split = 0  # 用比例切割
    kept_count = 0        # 太短不拆

    for idx, row in df.iterrows():
        src = str(row['transliteration'])
        tgt = str(row['translation'])

        # 拆分英文句子
        tgt_sents = [t.strip() for t in re.split(r'(?<=[.!?])\s+', tgt) if t.strip()]

        # 只有 1 句或太短 → 不拆
        if len(tgt_sents) <= 1 or len(src) < 20:
            kept_count += 1
            if len(src) > 3 and len(tgt) > 3:
                aligned_data.append({'transliteration': src, 'translation': tgt})
            continue

        # 方法 A：Akkadian 有換行且數量匹配 → 直接配對
        src_lines = [s.strip() for s in src.split('\n') if s.strip()]
        if len(src_lines) > 1 and len(src_lines) == len(tgt_sents):
            exact_split += 1
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3:
                    aligned_data.append({'transliteration': s, 'translation': t})
            continue

        # 方法 B：按英文句子長度比例切割 Akkadian
        proportional_split += 1
        tgt_lengths = [len(s) for s in tgt_sents]
        total_tgt_len = sum(tgt_lengths)

        # 計算每段 Akkadian 的切割位置
        src_text = src.strip()
        src_total = len(src_text)
        cut_positions = []
        cumulative = 0
        for length in tgt_lengths[:-1]:  # 最後一段不需要切割點
            cumulative += length
            raw_pos = int(src_total * cumulative / total_tgt_len)
            # 在空格處切割，避免切斷單字（往前後各找最近的空格）
            best_pos = raw_pos
            for offset in range(0, 20):
                if raw_pos + offset < src_total and src_text[raw_pos + offset] == ' ':
                    best_pos = raw_pos + offset
                    break
                if raw_pos - offset >= 0 and src_text[raw_pos - offset] == ' ':
                    best_pos = raw_pos - offset
                    break
            cut_positions.append(best_pos)

        # 切割 Akkadian
        cuts = [0] + cut_positions + [src_total]
        src_segments = [src_text[cuts[i]:cuts[i+1]].strip() for i in range(len(cuts)-1)]

        # 配對
        for s, t in zip(src_segments, tgt_sents):
            if len(s) > 3 and len(t) > 3:
                aligned_data.append({'transliteration': s, 'translation': t})

    print(f"=== Sentence Alignment ===")
    print(f"  {exact_split} docs: exact newline match")
    print(f"  {proportional_split} docs: proportional split")
    print(f"  {kept_count} docs: kept whole (single sentence or too short)")
    print(f"  {len(df)} docs -> {len(aligned_data)} training pairs")
    return pd.DataFrame(aligned_data)


# %%
# 執行資料擴增。
train_expanded = proportional_sentence_aligner(train_df)
print(f"Expanded Train Data: {len(train_expanded)} sentences (Alignment applied)")

# Gap 正規化：統一訓練資料中的破損標記
# 先記錄正規化前的狀態
before_src = train_expanded['transliteration'].copy()
before_tgt = train_expanded['translation'].copy()

train_expanded['transliteration'] = train_expanded['transliteration'].apply(normalize_gaps)
train_expanded['translation'] = train_expanded['translation'].apply(normalize_gaps)

# 診斷訊息：顯示正規化效果
src_changed = (before_src != train_expanded['transliteration']).sum()
tgt_changed = (before_tgt != train_expanded['translation']).sum()
src_gap_count = train_expanded['transliteration'].str.count('<gap>').sum()
src_biggap_count = train_expanded['transliteration'].str.count('<big_gap>').sum()
tgt_gap_count = train_expanded['translation'].str.count('<gap>').sum()
tgt_biggap_count = train_expanded['translation'].str.count('<big_gap>').sum()

print(f"=== Gap Normalization ===")
print(f"  Transliteration: {src_changed} rows changed, {int(src_gap_count)} <gap>, {int(src_biggap_count)} <big_gap>")
print(f"  Translation:     {tgt_changed} rows changed, {int(tgt_gap_count)} <gap>, {int(tgt_biggap_count)} <big_gap>")

# 顯示 3 筆前後對比範例（找到差異處附近顯示）
changed_idx = before_src[before_src != train_expanded['transliteration']].index
if len(changed_idx) > 0:
    print(f"\n  --- Sample changes (transliteration) ---")
    for i in changed_idx[:3]:
        b, a = before_src[i], train_expanded['transliteration'][i]
        # 找到第一個不同的位置，往前取 20 字元作為上下文
        diff_pos = next((j for j in range(min(len(b), len(a))) if b[j] != a[j]), 0)
        start = max(0, diff_pos - 20)
        print(f"  BEFORE: ...{b[start:start+80]}")
        print(f"  AFTER:  ...{a[start:start+80]}")
        print()

# 轉換為 Hugging Face Dataset 格式並分割為訓練集/驗證集。
dataset = Dataset.from_pandas(train_expanded)
# 以 test_size=0.1 建立驗證集。
split_datasets = dataset.train_test_split(test_size=0.1, seed=42)
# 分割後，鍵值為 'train' 和 'test'（我們將 'test' 作為驗證集使用）。


# %%
# ==========================================
# 3. 分詞與資料預處理
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

# 修正 dpc-starter-train 中對應的部分。
PREFIX = "translate Akkadian to English: "

def preprocess_function(examples):
    inputs = [PREFIX + str(ex) for ex in examples["transliteration"]]
    targets = [str(ex) for ex in examples["translation"]]
    
    model_inputs = tokenizer(inputs, max_length=Config.MAX_LENGTH, truncation=True)
    labels = tokenizer(targets, max_length=Config.MAX_LENGTH, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = split_datasets["train"].map(preprocess_function, batched=True)
tokenized_val = split_datasets["test"].map(preprocess_function, batched=True)

# === 截斷警告：檢查有多少筆被 MAX_LENGTH 截斷 ===
src_lengths = [len(tokenizer(PREFIX + str(ex))["input_ids"]) for ex in split_datasets["train"]["transliteration"][:500]]
tgt_lengths = [len(tokenizer(str(ex))["input_ids"]) for ex in split_datasets["train"]["translation"][:500]]
src_truncated = sum(1 for l in src_lengths if l > Config.MAX_LENGTH)
tgt_truncated = sum(1 for l in tgt_lengths if l > Config.MAX_LENGTH)
print(f"=== Truncation Check (sampled first 500) ===")
print(f"  Source: {src_truncated}/{len(src_lengths)} exceed MAX_LENGTH={Config.MAX_LENGTH} (max={max(src_lengths)}, avg={sum(src_lengths)//len(src_lengths)})")
print(f"  Target: {tgt_truncated}/{len(tgt_lengths)} exceed MAX_LENGTH={Config.MAX_LENGTH} (max={max(tgt_lengths)}, avg={sum(tgt_lengths)//len(tgt_lengths)})")
if src_truncated > len(src_lengths) * 0.1:
    print(f"  ⚠️ WARNING: >10% source truncated! Consider increasing MAX_LENGTH.")

# === 訓練樣本預覽：看幾組實際資料確認品質 ===
print(f"\n=== Training Sample Preview (3 pairs) ===")
print(f"  Train: {len(tokenized_train)}, Validation: {len(tokenized_val)}")
for i in range(min(3, len(split_datasets["train"]))):
    src = split_datasets["train"][i]["transliteration"][:100]
    tgt = split_datasets["train"][i]["translation"][:100]
    print(f"  [{i}] SRC: {src}")
    print(f"      TGT: {tgt}")
    print()


# %%
# ==========================================
# 4. 模型訓練（微調）
# ==========================================
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# === 模型資訊 ===
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"=== Model Info ===")
print(f"  Model: {Config.MODEL_NAME}")
print(f"  Total params: {total_params:,} ({total_params/1e6:.1f}M)")
print(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 評估指標（chrF++ 是競賽評估指標的一部分，衡量字元層級的精確度/重疊度）。
metric = evaluate.load("chrf")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]

    # ByT5 偶爾會產生超出有效 Unicode 範圍的 token ID，需要 clip 掉
    preds = np.where((preds < 0) | (preds >= tokenizer.vocab_size), tokenizer.pad_token_id, preds)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # 忽略標籤中的 -100。
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"chrf": result["score"]}

args = Seq2SeqTrainingArguments(
    output_dir=Config.OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=Config.LEARNING_RATE,

    # === byt5-base 記憶體優化 ===
    fp16=False,                     # ★byt5 用 fp16 容易 NaN，保持 fp32
    per_device_train_batch_size=2,  # ★base 模型更大，batch 降到 2
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # ★2×4=8 effective batch size
    # ======================

    weight_decay=0.01,
    save_total_limit=2,             # ★保留最近 2 個 checkpoint，確保最佳模型不被刪除
    save_only_model=True,           # ★只存模型權重 (~2.2GB/個)，節省空間
    num_train_epochs=Config.EPOCHS,
    predict_with_generate=True,
    logging_steps=50,
    report_to="none",

    lr_scheduler_type="cosine",              # cosine decay，訓練更穩定
    greater_is_better=True,                  # chrF 越高越好
    load_best_model_at_end=True,             # 訓練結束自動載入最佳 checkpoint
    metric_for_best_model="chrf",            # 用 chrF 選最佳模型
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Starting Training (FP32 mode)...")
trainer.train()


# %%
# --- 儲存模型 ---
# 重要：此處儲存的模型將在下一個筆記本中載入。
trainer.save_model(Config.OUTPUT_DIR)
tokenizer.save_pretrained(Config.OUTPUT_DIR)
print(f"Model saved to {Config.OUTPUT_DIR}")

