# %% [markdown]
# # Deep Past Initiative – Machine Translation (Training Notebook)
# 
# This notebook is a **starter / baseline** for this Kaggle competition.
# 
# Main ideas:
# - Use **ByT5** to handle noisy Akkadian transliterations at the character level
# - Perform **simple sentence alignment** to increase training data
# - Fine-tune using HuggingFace `Trainer`
# 
# 
# Inference Code is [here](https://www.kaggle.com/code/takamichitoda/dpc-starter-infer).

# %%
!pip install evaluate sacrebleu

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
    # Akkadian transliteration contains a lot of noise and many unknown words, so
    # ByT5, which processes text at the character (byte) level rather than the word level, is the strongest choice.
    MODEL_NAME = "google/byt5-small" 
    
    # ByT5 tends to produce longer token sequences, but 512 tokens is enough at the sentence level.
    MAX_LENGTH = 512
    
    BATCH_SIZE = 8       # Adjust depending on GPU memory (on a P100 you can usually go with 8–16).
    EPOCHS = 20
    LEARNING_RATE = 2e-4
    OUTPUT_DIR = "./byt5-akkadian-model"

# %%
# Fix the seed (for reproducibility).
def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
seed_everything()

# %%
INPUT_DIR = "/kaggle/input/deep-past-initiative-machine-translation"
train_df = pd.read_csv(f"{INPUT_DIR}/train.csv")
test_df = pd.read_csv(f"{INPUT_DIR}/test.csv")

# %%
print(f"Original Train Data: {len(train_df)} docs")

# %%
def simple_sentence_aligner(df):
    """
    【戦略の肝】
    Trainデータの「文書(複数文)」を、Testデータと同じ「文(1文)」に分割します。
    ここでは「英語の文数」と「アッカド語の行数」が一致する場合のみ分割する
    というヒューリスティック（簡易ルール）を使います。
    """
    aligned_data = []
    
    for idx, row in df.iterrows():
        src = str(row['transliteration'])
        tgt = str(row['translation'])
        
        # Split the English text by sentence-ending punctuation.
        tgt_sents = [t.strip() for t in re.split(r'(?<=[.!?])\s+', tgt) if t.strip()]
        
        # Assume the Akkadian text is often separated by newlines and split accordingly.
        src_lines = [s.strip() for s in src.split('\n') if s.strip()]
        
        # If the counts match, trust it as 1-to-1 pairs and use the split version.
        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3: # Remove junk/noisy data.
                    aligned_data.append({'transliteration': s, 'translation': t})
        else:
            # If splitting fails (counts don't match), keep the original document pair as-is (safe fallback).
            aligned_data.append({'transliteration': src, 'translation': tgt})
            
    return pd.DataFrame(aligned_data)


# %%
# Run data augmentation.
train_expanded = simple_sentence_aligner(train_df)
print(f"Expanded Train Data: {len(train_expanded)} sentences (Alignment applied)")

# Convert to Hugging Face Dataset format & split into Train/Val.
dataset = Dataset.from_pandas(train_expanded)
# Create a validation set with test_size=0.1.
split_datasets = dataset.train_test_split(test_size=0.1, seed=42)
# After splitting, the keys are 'train' and 'test' (we'll use 'test' as validation).


# %%
# ==========================================
# 3. Tokenization & preprocessing
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

# Fix the corresponding section in dpc-starter-train.
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


# %%
# ==========================================
# 4. Model training (fine-tuning)
# ==========================================
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Metric (chrF++ is part of the competition metric and measures character-level precision/overlap).
metric = evaluate.load("chrf")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Ignore -100 in the labels.
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
    
    # === Key fixes ===
    fp16=False,                     # ★Set to False to prevent a NaN error (required).
    per_device_train_batch_size=4,  # ★fp32 uses more memory, so reduce the batch size (8 -> 4).
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,  # ★To compensate, accumulate gradients to keep the effective batch size at 8.
    # ======================
    
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=Config.EPOCHS,
    predict_with_generate=True,
    logging_steps=10,               # Inspect logs in more detail.
    report_to="none"
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
# --- Save Model ---
# Important: the model saved here will be loaded in the next notebook.
trainer.save_model(Config.OUTPUT_DIR)
tokenizer.save_pretrained(Config.OUTPUT_DIR)
print(f"Model saved to {Config.OUTPUT_DIR}")



