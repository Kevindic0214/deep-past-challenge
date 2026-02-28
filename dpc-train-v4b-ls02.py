# %% [markdown]
# # Deep Past Initiative – 訓練 v4b：label_smoothing=0.2 變體
#
# v4b（與 v4 唯一差異：label_smoothing 從 0.1 改為 0.2，kh0a 參考腳本使用此值）：
# - label_smoothing_factor=0.2（較強的正則化）
# - warmup_steps=200（避免學習率一開始太激進）
# - EPOCHS=15（v2 的 10 epochs 可能欠擬合）
# - eval 每 3 epoch 一次（節省時間）
# - eval set 固定 300 筆（減少 eval 時間）
# - greedy eval（generation_num_beams=1，只看趨勢不需精準）
# - generation_max_length=512（ByT5 必須設定，否則輸出被截斷）
# - StderrLogCallback（Kaggle Logs 頁面可見進度）
#
# 與 v4 比較後，選擇分數更高的 label_smoothing 值用於後續階段
#

# %%
!pip install evaluate sacrebleu

# %%
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")

# %%
import gc
import re
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
)
import evaluate
import logging
import sys
import time

# 設定 logging 到 stderr（Kaggle Logs 頁面可見）
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class StderrLogCallback(TrainerCallback):
    """把訓練 metrics 寫到 stderr，讓 Kaggle Logs 頁面能即時看到進度"""
    def __init__(self):
        self.train_start_time = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        parts = []
        for k, v in logs.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        eta_str = ""
        if self.train_start_time and state.global_step > 0:
            elapsed = time.time() - self.train_start_time
            speed = state.global_step / elapsed
            remaining = (state.max_steps - state.global_step) / speed
            elapsed_h, elapsed_m = int(elapsed // 3600), int(elapsed % 3600 // 60)
            remain_h, remain_m = int(remaining // 3600), int(remaining % 3600 // 60)
            eta_str = f" | {elapsed_h}h{elapsed_m:02d}m elapsed, ~{remain_h}h{remain_m:02d}m left"
        logger.info(f"[Step {state.global_step}/{state.max_steps}] {', '.join(parts)}{eta_str}")

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()
        logger.info(f"Training started: {state.max_steps} total steps, {args.num_train_epochs} epochs")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
            logger.info(f"[Eval @ step {state.global_step}] {', '.join(parts)}")
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared after eval")

    def on_train_end(self, args, state, control, **kwargs):
        logger.info(f"Training finished. Total steps: {state.global_step}, Best metric: {state.best_metric}")


# %%
class Config:
    MODEL_NAME = "google/byt5-base"
    MAX_LENGTH = 512

    BATCH_SIZE = 8          # effective batch size（2 × 4 gradient accumulation）
    EPOCHS = 15             # v2 是 10，增加到 15 以防欠擬合
    LEARNING_RATE = 2e-4
    OUTPUT_DIR = "./byt5-base-akkadian-v4b"

# %%
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
    text = re.sub(r'\[\.\.\.\]', '<big_gap>', text)
    text = re.sub(r'\.{3,}', '<big_gap>', text)
    text = re.sub(r'…+', '<big_gap>', text)
    text = re.sub(r'xx+', '<gap>', text)
    text = re.sub(r'(?<=\s)x(?=\s)', '<gap>', text)
    text = re.sub(r'<gap>\s*<gap>', '<big_gap>', text)
    text = re.sub(r'<big_gap>\s*<big_gap>', '<big_gap>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def proportional_sentence_aligner(df):
    """
    將訓練資料的「文件」拆分成「句子對」，讓訓練格式更接近測試集（單句翻譯）。

    方法：
    1. 先用標點符號拆分英文翻譯成多個句子
    2. 如果 Akkadian 有換行符號且數量匹配 -> 直接配對
    3. 否則，按英文句子的字元長度比例，等比切割 Akkadian 文字
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

    print(f"=== Sentence Alignment ===")
    print(f"  {exact_split} docs: exact newline match")
    print(f"  {proportional_split} docs: proportional split")
    print(f"  {kept_count} docs: kept whole (single sentence or too short)")
    print(f"  {len(df)} docs -> {len(aligned_data)} training pairs")
    return pd.DataFrame(aligned_data)


# %%
train_expanded = proportional_sentence_aligner(train_df)
print(f"Expanded Train Data: {len(train_expanded)} sentences (Alignment applied)")

# Gap 正規化
before_src = train_expanded['transliteration'].copy()
before_tgt = train_expanded['translation'].copy()

train_expanded['transliteration'] = train_expanded['transliteration'].apply(normalize_gaps)
train_expanded['translation'] = train_expanded['translation'].apply(normalize_gaps)

src_changed = (before_src != train_expanded['transliteration']).sum()
tgt_changed = (before_tgt != train_expanded['translation']).sum()
src_gap_count = train_expanded['transliteration'].str.count('<gap>').sum()
src_biggap_count = train_expanded['transliteration'].str.count('<big_gap>').sum()
tgt_gap_count = train_expanded['translation'].str.count('<gap>').sum()
tgt_biggap_count = train_expanded['translation'].str.count('<big_gap>').sum()

print(f"=== Gap Normalization ===")
print(f"  Transliteration: {src_changed} rows changed, {int(src_gap_count)} <gap>, {int(src_biggap_count)} <big_gap>")
print(f"  Translation:     {tgt_changed} rows changed, {int(tgt_gap_count)} <gap>, {int(tgt_biggap_count)} <big_gap>")

changed_idx = before_src[before_src != train_expanded['transliteration']].index
if len(changed_idx) > 0:
    print(f"\n  --- Sample changes (transliteration) ---")
    for i in changed_idx[:3]:
        b, a = before_src[i], train_expanded['transliteration'][i]
        diff_pos = next((j for j in range(min(len(b), len(a))) if b[j] != a[j]), 0)
        start = max(0, diff_pos - 20)
        print(f"  BEFORE: ...{b[start:start+80]}")
        print(f"  AFTER:  ...{a[start:start+80]}")
        print()

# %%
# ==========================================
# 資料集分割：固定 300 筆 eval set（減少 eval 時間）
# ==========================================
dataset = Dataset.from_pandas(train_expanded)
split_datasets = dataset.train_test_split(test_size=300, seed=42)

# %%
# ==========================================
# 分詞與資料預處理
# ==========================================
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
print(f"=== Truncation Check (sampled first 500) ===")
print(f"  Source: {src_truncated}/{len(src_lengths)} exceed MAX_LENGTH={Config.MAX_LENGTH} (max={max(src_lengths)}, avg={sum(src_lengths)//len(src_lengths)})")
print(f"  Target: {tgt_truncated}/{len(tgt_lengths)} exceed MAX_LENGTH={Config.MAX_LENGTH} (max={max(tgt_lengths)}, avg={sum(tgt_lengths)//len(tgt_lengths)})")
if src_truncated > len(src_lengths) * 0.1:
    print(f"  WARNING: >10% source truncated! Consider increasing MAX_LENGTH.")

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
# 模型訓練
# ==========================================
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"=== Model Info ===")
print(f"  Model: {Config.MODEL_NAME}")
print(f"  Total params: {total_params:,} ({total_params/1e6:.1f}M)")
print(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

metric = evaluate.load("chrf")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]

    preds = np.where((preds < 0) | (preds >= tokenizer.vocab_size), tokenizer.pad_token_id, preds)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"chrf": result["score"]}

# 計算 eval 頻率：每 3 epoch eval 一次
PER_DEVICE_BATCH = 2
GRAD_ACCUM = 4
steps_per_epoch = len(tokenized_train) // (PER_DEVICE_BATCH * GRAD_ACCUM)
eval_every_n_epochs = 3
eval_steps = steps_per_epoch * eval_every_n_epochs

print(f"\n=== Training Plan ===")
print(f"  Steps per epoch: ~{steps_per_epoch}")
print(f"  Total steps: ~{steps_per_epoch * Config.EPOCHS}")
print(f"  Warmup steps: 200")
print(f"  Eval every: {eval_every_n_epochs} epochs ({eval_steps} steps), eval set: 300 samples")
print(f"  Label smoothing: 0.2")

args = Seq2SeqTrainingArguments(
    output_dir=Config.OUTPUT_DIR,

    # v4: 每 3 epoch eval 一次（省時間）
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=eval_steps,

    learning_rate=Config.LEARNING_RATE,

    # v4 新增：label smoothing + warmup
    label_smoothing_factor=0.2,
    warmup_steps=200,

    fp16=False,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=GRAD_ACCUM,

    # v4: greedy eval（省時間）+ 確保 ByT5 輸出不被截斷
    generation_max_length=Config.MAX_LENGTH,
    generation_num_beams=1,

    weight_decay=0.01,
    save_total_limit=3,
    save_only_model=True,
    num_train_epochs=Config.EPOCHS,
    predict_with_generate=True,
    logging_steps=50,
    report_to="none",

    lr_scheduler_type="cosine",
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
    compute_metrics=compute_metrics,
    callbacks=[StderrLogCallback()]
)

logger.info("Starting Training (v4b: label_smoothing=0.2, warmup=200, 15 epochs)...")
trainer.train()


# %%
trainer.save_model(Config.OUTPUT_DIR)
tokenizer.save_pretrained(Config.OUTPUT_DIR)
print(f"Model saved to {Config.OUTPUT_DIR}")
