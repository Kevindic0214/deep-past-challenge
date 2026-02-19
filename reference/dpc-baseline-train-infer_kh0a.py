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
# Original train  : https://www.kaggle.com/code/takamichitoda/dpc-starter-train, fixed metric calculation, adding API for uploading model as kaggle dataset
# 
# Best checkpoint : https://www.kaggle.com/datasets/llkh0a/byt5-akkadian-model is resume train from 
# 
# this checkpoint : https://www.kaggle.com/datasets/jeanjean111/byt5-base-big-data2

# %% [markdown]
# # Configuration

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
    """Configuration for ByT5 Akkadian-to-English Machine Translation"""
    # ====== Model Configuration ======
    MODEL_NAME = "google/byt5-base"
    PRETRAINED_DIR = None  # Set to a path to resume training; None to start from scratch
    PRETRAINED_DIR = "/kaggle/input/byt5-akkadian-model"
    MAX_LENGTH = 512
    
    # ====== Training Configuration ======
    INPUT_DIR = "/kaggle/input/deep-past-initiative-machine-translation"
    SEED = 42
    EPOCHS = 1
    LEARNING_RATE = 1e-4
    TRAIN_BATCH_SIZE = 2    # per_device_train_batch_size
    EVAL_BATCH_SIZE = 2     # per_device_eval_batch_size
    GRADIENT_ACCUMULATION_STEPS = 8
    OUTPUT_DIR = "./byt5-akkadian-model"
    
    # ====== Inference & Upload Configuration ======
    MAKE_SUBMISSION = False   # Set to True to run inference on test set
    UPLOAD_MODEL = False     # Set to True to upload model to Kaggle
    KAGGLE_USERNAME = "<redacted>"
    KAGGLE_KEY = "<redacted>"
    DATASET_NAME = "byt5-akkadian-model-private"

# %%
# Fix the seed (for reproducibility).
def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

seed_everything(Config.SEED)

# %%
INPUT_DIR = Config.INPUT_DIR
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
def create_bidirectional_data(dataset_split):
    df = dataset_split.to_pandas()
    
    # 方向1: Akkadian -> English (元のタスク)
    df_fwd = df.copy()
    df_fwd['input_text'] = "translate Akkadian to English: " + df_fwd['transliteration'].astype(str)
    df_fwd['target_text'] = df_fwd['translation'].astype(str)
    
    # 方向2: English -> Akkadian (逆翻訳タスク)
    df_bwd = df.copy()
    df_bwd['input_text'] = "translate English to Akkadian: " + df_bwd['translation'].astype(str)
    df_bwd['target_text'] = df_bwd['transliteration'].astype(str)
    
    # 結合してシャッフル
    df_combined = pd.concat([df_fwd, df_bwd], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return Dataset.from_pandas(df_combined)

def create_unidirectional_data(dataset_split):
    # Validation用: 形式だけ揃える (Akkadian -> Englishのみ)
    df = dataset_split.to_pandas()
    df['input_text'] = "translate Akkadian to English: " + df['transliteration'].astype(str)
    df['target_text'] = df['translation'].astype(str)
    return Dataset.from_pandas(df)

# Trainデータは双方向化 (データ数が2倍になります)
bidirectional_train = create_bidirectional_data(split_datasets["train"])

# Validationデータは単方向のまま (評価のため)
unidirectional_val = create_unidirectional_data(split_datasets["test"])

print(f"Train samples: {len(bidirectional_train)} (Bidirectional)")
print(f"Val samples:   {len(unidirectional_val)} (Unidirectional)")

# %%
# ==========================================
# 3. Tokenization & preprocessing
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

def preprocess_function(examples):
    # データ作成時にPrefix込みの "input_text" と "target_text" を作っているのでそれを使う
    inputs = [str(ex) for ex in examples["input_text"]]
    targets = [str(ex) for ex in examples["target_text"]]
    
    model_inputs = tokenizer(inputs, max_length=Config.MAX_LENGTH, truncation=True)
    labels = tokenizer(targets, max_length=Config.MAX_LENGTH, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 作成した新しいデータセットに対してmapを適用
tokenized_train = bidirectional_train.map(preprocess_function, batched=True)
tokenized_val = unidirectional_val.map(preprocess_function, batched=True)

# %%
from transformers import TrainerCallback

DEBUG_EVAL_PRINTED = True

class PrintMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = getattr(state, "epoch", None)
        # find the most recent training loss from the log history
        train_loss = None
        for entry in reversed(state.log_history):
            if "loss" in entry:
                train_loss = entry["loss"]
                break
        prefix = f"[Epoch {epoch}] " if epoch is not None else ""
        # if train_loss is not None:
        #     print(prefix + f"Train Loss={train_loss:.4f}")
        if metrics:
            keys = ["eval_loss", "eval_chrf", "eval_bleu", "eval_geo_mean"]
            metric_items = ", ".join(f"{k}={metrics[k]:.4f}" for k in keys if k in metrics)
            print(prefix + f"Train Loss={train_loss:.4f}, "+metric_items)

# %% [markdown]
# # Load Pretrained Model (Optional)

# %%
# ==========================================
# 4. Model training (fine-tuning)
# ==========================================
gc.collect()
torch.cuda.empty_cache()


# %%
gc.collect()
torch.cuda.empty_cache()

# Load model based on PRETRAINED_DIR
if Config.PRETRAINED_DIR is not None:
    print(f"Loading pretrained model from {Config.PRETRAINED_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(
        Config.PRETRAINED_DIR,
        local_files_only=True,
        use_fast=False,  # byt5 uses a byte-level tokenizer; safe to set use_fast=False
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        Config.PRETRAINED_DIR,
        local_files_only=True,
    )
    print("✓ Pretrained model loaded successfully")
else:
    print(f"Training from scratch using {Config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME)
    print("✓ Base model loaded successfully")

# %%
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 評価指標の準備
metric_chrf = evaluate.load("chrf")
metric_bleu = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Seq2SeqTrainer/Trainer の返り値が tuple になることがあるので保険
    if isinstance(preds, tuple):
        preds = preds[0]

    # ★ここが重要★
    # preds が logits の場合: (batch, seq, vocab) になりがち → argmax で token ids に変換
    if hasattr(preds, "ndim") and preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    # int 化 & 範囲外対策（ByT5 の decode が chr を使うため）
    preds = preds.astype(np.int64)
    preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # === Fix: Replace -100 in labels before decoding ===
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ===================================================

    # Wrap references in list of lists for both metrics to be safe
    formatted_refs = [[x] for x in decoded_labels]

    chrf = metric_chrf.compute(predictions=decoded_preds, references=formatted_refs)["score"]
    bleu = metric_bleu.compute(predictions=decoded_preds, references=formatted_refs)["score"]
    geo_mean = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0
    
    return {"chrf": chrf, "bleu": bleu, "geo_mean": geo_mean}

args = Seq2SeqTrainingArguments(
    output_dir=Config.OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=Config.LEARNING_RATE,
    optim="adafactor",
    label_smoothing_factor=0.2,
    
    # === Key fixes ===
    fp16=False,                     # ★Set to False to prevent a NaN error (required).
    per_device_train_batch_size=Config.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=Config.EVAL_BATCH_SIZE,
    gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
    # ======================
    generation_max_length=Config.MAX_LENGTH,  # <--- Add this (512). Crucial for ByT5.
    generation_num_beams=2, 
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=Config.EPOCHS,
    predict_with_generate=True,
    logging_steps=10,               # Inspect logs in more detail.
    report_to="none",
    # show/use competition_metric as the main criterion
    load_best_model_at_end=True,
    metric_for_best_model="geo_mean",
    greater_is_better=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[PrintMetricsCallback()]
)

print("Starting Training (FP32 mode)...")
trainer.train()

# %%
# --- Save Model ---
# Important: the model saved here will be loaded in the next notebook.
trainer.save_model(Config.OUTPUT_DIR)
tokenizer.save_pretrained(Config.OUTPUT_DIR)
print(f"Model saved to {Config.OUTPUT_DIR}")

# %% [markdown]
# # Upload Model via Kaggle API

# %%
# === Upload best model to Kaggle Dataset (Optional) ===
# Only upload if UPLOAD_MODEL is True

if not Config.UPLOAD_MODEL:
    print("Model upload disabled (Config.UPLOAD_MODEL = False)")
else:
    import json
    import os
    import subprocess
    from pathlib import Path
    import getpass
    from datetime import datetime
    
    # 1. Setup Credentials
    kaggle_api_token = {
        "username": Config.KAGGLE_USERNAME, 
        "key": Config.KAGGLE_KEY
    }
    
    # Write to default location ~/.kaggle/kaggle.json
    p = Path.home() / ".kaggle"
    p.mkdir(exist_ok=True)
    k = p / "kaggle.json"
    k.write_text(json.dumps(kaggle_api_token))
    try:
        k.chmod(0o600)
    except Exception:
        pass  # chmod may not work on Windows
    
    # Also set env vars for the current process
    os.environ["KAGGLE_USERNAME"] = kaggle_api_token["username"]
    os.environ["KAGGLE_KEY"] = kaggle_api_token["key"]

    def ensure_kaggle_creds(interactive=True):
        """Ensure Kaggle credentials exist. If not and interactive==True, prompt the user to enter them."""
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.exists():
            try:
                with open(kaggle_json, "r") as f:
                    data = json.load(f)
                os.environ["KAGGLE_USERNAME"] = data.get("username", "")
                os.environ["KAGGLE_KEY"] = data.get("key", "")
                return True
            except Exception as e:
                print("Failed to read existing ~/.kaggle/kaggle.json:", e)

        if not interactive:
            print("No ~/.kaggle/kaggle.json found and interactive mode is off.")
            return False

        print("No ~/.kaggle/kaggle.json found. Please enter Kaggle credentials to create it.")
        username = input("Kaggle username: ").strip()
        key = getpass.getpass("Kaggle API key (hidden): ").strip()

        if username and key:
            kaggle_json.parent.mkdir(parents=True, exist_ok=True)
            with open(kaggle_json, "w") as f:
                json.dump({"username": username, "key": key}, f)
            try:
                kaggle_json.chmod(0o600)
            except Exception:
                pass
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key
            print("Credentials saved to ~/.kaggle/kaggle.json")
            return True
        else:
            print("No credentials provided. Skipping upload.")
            return False

    if ensure_kaggle_creds(interactive=False):
        model_path = Path(Config.OUTPUT_DIR).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Saved model directory not found: {model_path}. Run training and save the model first.")

        # Ensure .kaggleignore doesn't exclude files
        ignore_file = model_path / ".kaggleignore"
        if ignore_file.exists():
            ignore_file.unlink()

        # Prepare minimal metadata
        username = os.environ.get("KAGGLE_USERNAME") or "your-username"
        dataset_id = f"{username}/{Config.DATASET_NAME}"
        
        # Create detailed description with timestamp and config
        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        description = (
            f"ByT5 fine-tuned model for DPC Akkadian->English.\n"
            f"Upload Time: {upload_time}\n"
            f"Epochs: {Config.EPOCHS}, LR: {Config.LEARNING_RATE}, Seed: {Config.SEED}\n"
            f"Training Batch: {Config.TRAIN_BATCH_SIZE}, Eval Batch: {Config.EVAL_BATCH_SIZE}\n"
            f"Base Model: {Config.MODEL_NAME}"
        )
        
        metadata = {
            "title": Config.DATASET_NAME,
            "id": dataset_id,
            "description": description,
            "licenses": [{"name": "CC0-1.0"}],
        }

        # Write metadata
        metadata_file = model_path / "dataset-metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Uploading {model_path} as dataset {dataset_id} (this may take a while)...")

        # Try create; if exists, create a new version
        create = subprocess.run(["kaggle", "datasets", "create", "-p", str(model_path), "--dir-mode", "zip"], capture_output=True, text=True)
        if create.returncode != 0:
            stderr_l = (create.stderr or "").lower()
            print("Create stderr:", create.stderr)
            if "already exists" in stderr_l or "dataset with slug" in stderr_l:
                print("Dataset already exists. Creating a new version...")
                create = subprocess.run([
                    "kaggle", "datasets", "version", "-p", str(model_path), "-m", "New model version", "--dir-mode", "zip", "--force"
                ], capture_output=True, text=True)

        print("=== Kaggle CLI stdout ===")
        print(create.stdout)
        print("=== Kaggle CLI stderr ===")
        print(create.stderr)

        if create.returncode == 0:
            print("✅ Upload successful.")
        else:
            print("❌ Upload failed. Inspect the error above.")
    else:
        print("Skipping upload due to missing credentials.")

# %% [markdown]
# # Inference

# %%
if Config.MAKE_SUBMISSION:
    print("Starting Inference...")
    from torch.utils.data import DataLoader, Dataset
    
    PREFIX = "translate Akkadian to English: "

    class InferenceDataset(Dataset):
        def __init__(self, df, tokenizer, max_length=Config.MAX_LENGTH):
            self.texts = df['transliteration'].astype(str).tolist()
            self.texts = [PREFIX + i for i in self.texts]
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            inputs = self.tokenizer(
                text, 
                max_length=self.max_length, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0)
            }

    test_dataset = InferenceDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=Config.EVAL_BATCH_SIZE, shuffle=False)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE)
    model.eval()
    model.float()  # force weights to FP32

    all_predictions = []
    torch.set_grad_enabled(False)

    with torch.inference_mode():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=Config.MAX_LENGTH,
                num_beams=4,
                early_stopping=True,
            )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_predictions.extend([d.strip() for d in decoded])

    # Create submission file
    submission = pd.DataFrame({
        "id": test_df["id"],
        "translation": all_predictions
    })

    submission["translation"] = submission["translation"].apply(lambda x: x if len(x) > 0 else "broken text")
    
    submission.to_csv("submission.csv", index=False)
    print("✅ Submission file saved successfully!")
    print(submission.head())
else:
    print("Inference disabled (Config.MAKE_SUBMISSION = False)")


