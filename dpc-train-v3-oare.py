# %% [markdown]
# # Deep Past Initiative – 訓練 v3：Sentences_Oare 擴充版
# #
# 改進內容：
# - 加入 Sentences_Oare 的 8,044 筆句子級翻譯對（6,900 筆全新資料）
# - 結合原始 train.csv 的比例切割資料
# - 訓練資料量約 2 倍提升：~8,300 → ~16,000+
# #

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
import evaluate

# %%
class Config:
    MODEL_NAME = "google/byt5-base"
    MAX_LENGTH = 512
    BATCH_SIZE = 8       # effective batch size（透過 gradient accumulation 達成）
    EPOCHS = 10          # 資料量增加但保持 10 epochs，讓模型充分學習新資料
    LEARNING_RATE = 2e-4
    OUTPUT_DIR = "./byt5-base-akkadian-v3"

# %%
def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

seed_everything()

# %%
# ==========================================
# 1. 載入資料
# ==========================================
INPUT_DIR = "/kaggle/input/competitions/deep-past-initiative-machine-translation" if os.path.exists("/kaggle/input/competitions/deep-past-initiative-machine-translation") else "./deep-past-initiative-machine-translation"

# 原始訓練資料
train_df = pd.read_csv(f"{INPUT_DIR}/train.csv")
print(f"Original Train Data: {len(train_df)} docs")

# Sentences_Oare 擴充資料
# 優先從 Kaggle input 載入，否則從本地載入
OARE_PATHS = [
    "/kaggle/input/datasets/kevindic0214/sentences-oare-pairs/sentences_oare_pairs.csv",
    "./sentences_oare_pairs.csv",
]
oare_df = None
for path in OARE_PATHS:
    if os.path.exists(path):
        oare_df = pd.read_csv(path)
        print(f"Sentences_Oare loaded from: {path} ({len(oare_df)} pairs)")
        break
if oare_df is None:
    print("WARNING: sentences_oare_pairs.csv not found! Using train.csv only.")

# %%
# ==========================================
# 2. Gap 正規化
# ==========================================
def normalize_gaps(text):
    """統一各種破損/缺失標記為 <gap> 和 <big_gap>"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\[\.\.\.\]', '<big_gap>', text)
    text = re.sub(r'\.{3,}', '<big_gap>', text)
    text = re.sub(r'…+', '<big_gap>', text)
    text = re.sub(r'xx+', '<gap>', text)
    text = re.sub(r'(?<=\s)x(?=\s)', '<gap>', text)
    text = re.sub(r'<gap>\s*<gap>', '<big_gap>', text)
    text = re.sub(r'<big_gap>\s*<big_gap>', '<big_gap>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# %%
# ==========================================
# 3. 句子對齊（train.csv 的比例切割）
# ==========================================
def proportional_sentence_aligner(df):
    """
    將 train.csv 的文件拆成句子對。
    方法 A：Akkadian 換行符數量匹配英文句子數 → 直接配對
    方法 B：按英文句子長度比例切割 Akkadian
    """
    aligned_data = []
    exact_split = 0
    proportional_split = 0
    kept_count = 0

    for idx, row in df.iterrows():
        src = str(row['transliteration'])
        tgt = str(row['translation'])

        tgt_sents = [t.strip() for t in re.split(r'(?<=[.!?])\s+', tgt) if t.strip()]

        if len(tgt_sents) <= 1 or len(src) < 20:
            kept_count += 1
            if len(src) > 3 and len(tgt) > 3:
                aligned_data.append({'transliteration': src, 'translation': tgt})
            continue

        src_lines = [s.strip() for s in src.split('\n') if s.strip()]
        if len(src_lines) > 1 and len(src_lines) == len(tgt_sents):
            exact_split += 1
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3:
                    aligned_data.append({'transliteration': s, 'translation': t})
            continue

        proportional_split += 1
        tgt_lengths = [len(s) for s in tgt_sents]
        total_tgt_len = sum(tgt_lengths)
        src_text = src.strip()
        src_total = len(src_text)
        cut_positions = []
        cumulative = 0
        for length in tgt_lengths[:-1]:
            cumulative += length
            raw_pos = int(src_total * cumulative / total_tgt_len)
            best_pos = raw_pos
            for offset in range(0, 20):
                if raw_pos + offset < src_total and src_text[raw_pos + offset] == ' ':
                    best_pos = raw_pos + offset
                    break
                if raw_pos - offset >= 0 and src_text[raw_pos - offset] == ' ':
                    best_pos = raw_pos - offset
                    break
            cut_positions.append(best_pos)

        cuts = [0] + cut_positions + [src_total]
        src_segments = [src_text[cuts[i]:cuts[i+1]].strip() for i in range(len(cuts)-1)]

        for s, t in zip(src_segments, tgt_sents):
            if len(s) > 3 and len(t) > 3:
                aligned_data.append({'transliteration': s, 'translation': t})

    print(f"=== Sentence Alignment (train.csv) ===")
    print(f"  {exact_split} docs: exact newline match")
    print(f"  {proportional_split} docs: proportional split")
    print(f"  {kept_count} docs: kept whole")
    print(f"  {len(df)} docs -> {len(aligned_data)} pairs")
    return pd.DataFrame(aligned_data)

# %%
# ==========================================
# 4. 合併所有訓練資料
# ==========================================
print("\n=== Building combined dataset ===")

# Part A: train.csv 的句子級拆分
train_split = proportional_sentence_aligner(train_df)
print(f"  Part A (train.csv split): {len(train_split)} pairs")

# Part B: Sentences_Oare 的句子級翻譯
if oare_df is not None:
    oare_pairs = oare_df[['transliteration', 'translation']].copy()
    print(f"  Part B (Sentences_Oare): {len(oare_pairs)} pairs")
    print(f"    - New texts: {len(oare_df[oare_df['source'] == 'new'])} pairs")
    print(f"    - Train splits: {len(oare_df[oare_df['source'] == 'train_split'])} pairs")
else:
    oare_pairs = pd.DataFrame(columns=['transliteration', 'translation'])
    print(f"  Part B (Sentences_Oare): 0 pairs (not available)")

# 合併
combined = pd.concat([train_split, oare_pairs], ignore_index=True)
print(f"\n  Combined before cleanup: {len(combined)} pairs")

# Gap 正規化
combined['transliteration'] = combined['transliteration'].apply(normalize_gaps)
combined['translation'] = combined['translation'].apply(normalize_gaps)

# 去除完全重複
before = len(combined)
combined = combined.drop_duplicates(subset=['transliteration', 'translation'])
print(f"  After dedup: {len(combined)} pairs (removed {before - len(combined)} duplicates)")

# 過濾太短的
before = len(combined)
combined = combined[
    (combined['transliteration'].str.len() >= 5) &
    (combined['translation'].str.len() >= 3)
]
print(f"  After length filter: {len(combined)} pairs (removed {before - len(combined)})")

# Gap 正規化統計
gap_count = combined['transliteration'].str.count('<gap>').sum()
biggap_count = combined['transliteration'].str.count('<big_gap>').sum()
print(f"\n=== Gap Normalization ===")
print(f"  Transliteration: {int(gap_count)} <gap>, {int(biggap_count)} <big_gap>")

# 資料統計
print(f"\n=== Final Dataset Stats ===")
print(f"  Total pairs: {len(combined)}")
print(f"  Avg src length: {combined['transliteration'].str.len().mean():.1f} chars")
print(f"  Avg tgt length: {combined['translation'].str.len().mean():.1f} chars")
print(f"  Median src length: {combined['transliteration'].str.len().median():.0f} chars")
print(f"  Median tgt length: {combined['translation'].str.len().median():.0f} chars")

# %%
# ==========================================
# 5. 分詞與資料預處理
# ==========================================
dataset = Dataset.from_pandas(combined[['transliteration', 'translation']].reset_index(drop=True))
split_datasets = dataset.train_test_split(test_size=0.1, seed=42)

tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

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

# 截斷檢查
src_lengths = [len(tokenizer(PREFIX + str(ex))["input_ids"]) for ex in split_datasets["train"]["transliteration"][:500]]
tgt_lengths = [len(tokenizer(str(ex))["input_ids"]) for ex in split_datasets["train"]["translation"][:500]]
src_truncated = sum(1 for l in src_lengths if l > Config.MAX_LENGTH)
tgt_truncated = sum(1 for l in tgt_lengths if l > Config.MAX_LENGTH)
print(f"\n=== Truncation Check (sampled first 500) ===")
print(f"  Source: {src_truncated}/{len(src_lengths)} exceed MAX_LENGTH={Config.MAX_LENGTH}")
print(f"  Target: {tgt_truncated}/{len(tgt_lengths)} exceed MAX_LENGTH={Config.MAX_LENGTH}")

print(f"\n=== Training Sample Preview ===")
print(f"  Train: {len(tokenized_train)}, Validation: {len(tokenized_val)}")
for i in range(min(3, len(split_datasets["train"]))):
    src = split_datasets["train"][i]["transliteration"][:100]
    tgt = split_datasets["train"][i]["translation"][:100]
    print(f"  [{i}] SRC: {src}")
    print(f"      TGT: {tgt}")

# %%
# ==========================================
# 6. 模型訓練
# ==========================================
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n=== Model Info ===")
print(f"  Model: {Config.MODEL_NAME}")
print(f"  Total params: {total_params:,} ({total_params/1e6:.1f}M)")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

metric = evaluate.load("chrf")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]

    # ByT5 偶爾會產生超出有效 Unicode 範圍的 token ID
    preds = np.where((preds < 0) | (preds >= tokenizer.vocab_size), tokenizer.pad_token_id, preds)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"chrf": result["score"]}

# 計算訓練步數
steps_per_epoch = len(tokenized_train) // (2 * 4)  # per_device_batch=2, grad_accum=4
total_steps = steps_per_epoch * Config.EPOCHS
warmup_steps = min(500, total_steps // 10)
print(f"\n=== Training Plan ===")
print(f"  Steps per epoch: ~{steps_per_epoch}")
print(f"  Total steps: ~{total_steps}")
print(f"  Warmup steps: {warmup_steps}")

args = Seq2SeqTrainingArguments(
    output_dir=Config.OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=Config.LEARNING_RATE,

    fp16=False,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,

    weight_decay=0.01,
    save_total_limit=2,
    save_only_model=True,
    num_train_epochs=Config.EPOCHS,
    predict_with_generate=True,
    logging_steps=50,
    report_to="none",

    lr_scheduler_type="cosine",
    warmup_steps=warmup_steps,
    greater_is_better=True,
    load_best_model_at_end=True,
    metric_for_best_model="chrf",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

print("\nStarting Training...")
trainer.train()

# %%
trainer.save_model(Config.OUTPUT_DIR)
tokenizer.save_pretrained(Config.OUTPUT_DIR)
print(f"Model saved to {Config.OUTPUT_DIR}")


