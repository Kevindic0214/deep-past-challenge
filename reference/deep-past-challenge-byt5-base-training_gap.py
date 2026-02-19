# %%
# 注意：執行此腳本前，請先在終端機執行：pip install evaluate sacrebleu

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
    MODEL_NAME = "google/byt5-base" 
    
    # ByT5 tends to produce longer token sequences, but 512 tokens is enough at the sentence level.
    MAX_LENGTH = 512
    
    BATCH_SIZE = 8       # Adjust depending on GPU memory (on a P100 you can usually go with 8–16).
    EPOCHS = 10
    LEARNING_RATE = 2e-4
    OUTPUT_DIR = "./byt5-base-akkadian_gap3"

# %%
# class Config:
#     # Akkadian transliteration contains a lot of noise and many unknown words, so
#     # ByT5, which processes text at the character (byte) level rather than the word level, is the strongest choice.
#     MODEL_NAME = "/kaggle/input/deep-past-challenge-byt5-base-training/byt5-base-akkadian/" 
    
#     # ByT5 tends to produce longer token sequences, but 512 tokens is enough at the sentence level.
#     MAX_LENGTH = 512
    
#     BATCH_SIZE = 8       # Adjust depending on GPU memory (on a P100 you can usually go with 8–16).
#     EPOCHS = 3
#     LEARNING_RATE = 1e-5
#     OUTPUT_DIR = "./byt5-base-akkadian-continue-train2"

# %%
# Fix the seed (for reproducibility).
def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
seed_everything()

# %%
INPUT_DIR = "/kaggle/input/deep-past-initiative-machine-translation"
train_df = pd.read_csv(f"/kaggle/input/tain-gap-1227-5/output_gap_big_gap5.csv")
test_df = pd.read_csv(f"{INPUT_DIR}/test.csv")
print("ok")

# %%
# INPUT_DIR = "/kaggle/input/deep-past-initiative-machine-translation"
# #train_df = pd.read_csv(f"/kaggle/input/jifang-modify-1223-end/train (1).csv")
# train_df = pd.read_csv(f"/kaggle/input/setence-14-1225-train-2/train.csv")
# test_df = pd.read_csv(f"{INPUT_DIR}/test.csv")
# print("ok")

# %%
# INPUT_DIR = "/kaggle/input/deep-past-initiative-machine-translation"
# temp_df=pd.read_csv(f"/kaggle/input/new-train-tablet-19/published_texts_train_tablet.csv")
# temp_df.columns = ["oare_id","transliteration","translation"]
# train_df_temp = pd.read_csv(f"{INPUT_DIR}/train.csv")
# train_df = pd.concat([temp_df, train_df_temp], axis=0, ignore_index=True)

# test_df = pd.read_csv(f"{INPUT_DIR}/test.csv")

# %%
# INPUT_DIR = "/kaggle/input/deep-past-initiative-machine-translation"
# temp_df=pd.read_csv(f"/kaggle/input/old-assyrian-extended-corpus/akkadian_corpus.csv")
# temp_df = temp_df[temp_df['has_translation'] == True]

# temp_df= temp_df[["oare_id","transliteration","translation"]]
# train_df_temp = pd.read_csv(f"{INPUT_DIR}/train.csv")
# train_df = pd.concat([temp_df, train_df_temp], axis=0, ignore_index=True)
# test_df = pd.read_csv(f"{INPUT_DIR}/test.csv")

# %%
# INPUT_DIR = "/kaggle/input/deep-past-initiative-machine-translation"
# train_df = pd.read_csv(f"/kaggle/input/old-assyrian-extended-corpus/akkadian_corpus.csv")
# train_df = train_df[train_df['has_translation'] == True]
# print(train_df.shape)

# train_df['length_ratio'] = pd.to_numeric(train_df['length_ratio'], errors='coerce')
# train_df = train_df.dropna(subset=['length_ratio'])
# train_df = train_df[train_df['length_ratio'] >= 1]

# test_df = pd.read_csv(f"{INPUT_DIR}/test.csv")
# print(train_df.shape)


# %% [markdown]
# (1561, 17)
# (1127, 17)

# %%
print(f"Original Train Data: {len(train_df)} docs")

# %%
def simple_sentence_aligner(df):
    aligned_data = []
    
    for idx, row in df.iterrows():
        src = str(row['transliteration'])
        tgt = str(row['translation'])
        
        tgt_sents = [t.strip() for t in re.split(r'(?<=[.!?])\s+', tgt) if t.strip()]
        
        src_lines = [s.strip() for s in src.split('\n') if s.strip()]
        
        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3: 
                    aligned_data.append({'transliteration': s, 'translation': t})
        else:
            aligned_data.append({'transliteration': src, 'translation': tgt})
            
    return pd.DataFrame(aligned_data)


# %%

train_expanded = simple_sentence_aligner(train_df)
print(f"Expanded Train Data: {len(train_expanded)} sentences (Alignment applied)")

dataset = Dataset.from_pandas(train_expanded)
split_datasets = dataset.train_test_split(test_size=0.1, seed=42)

# %%
# ==========================================
# 3. Tokenization & preprocessing
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

# Fix the corresponding section in dpc-starter-train.
PREFIX = "translate Akkadian to English: "

def preprocess_function(examples):
    inputs = [PREFIX + str(ex) for ex in examples["transliteration"]]
    #inputs=inputs[:2]
    targets = [str(ex) for ex in examples["translation"]]
    #targets=targets[:2]
    #print(targets)
    # if len(inputs) <= 3:
    #     for i, (inp, tgt) in enumerate(zip(inputs, targets)):
    #         inp_tokens = tokenizer(inp, max_length=Config.MAX_LENGTH, truncation=True)["input_ids"]
    #         tgt_tokens = tokenizer(tgt, max_length=Config.MAX_LENGTH, truncation=True)["input_ids"]
    #         print(f"示例{i+1} - 输入token长度：{len(inp_tokens)} | 输出token长度：{len(tgt_tokens)}")


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
model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME, device_map='auto')
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
    save_strategy="no",
    learning_rate=Config.LEARNING_RATE,
    
    # === Key fixes ===
    fp16=False,                     # ★Set to False to prevent a NaN error (required).
    per_device_train_batch_size=2,  # ★fp32 uses more memory, so reduce the batch size (8 -> 4).
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,  # ★To compensate, accumulate gradients to keep the effective batch size at 8.
    # ======================
    
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=Config.EPOCHS,
    predict_with_generate=True,
    logging_steps=100,               # Inspect logs in more detail.
    report_to="none",
    
    # greater_is_better=True,
    # load_best_model_at_end=True,
    # metric_for_best_model="chrf",
    lr_scheduler_type="cosine",
    
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


