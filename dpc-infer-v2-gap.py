# %% [markdown]
# # 深度過去計畫 – 機器翻譯（推論筆記本）
# 
# 這個筆記本是此 Kaggle 競賽的**入門/基準**。
# 
# 訓練程式碼在[這裡](https://www.kaggle.com/code/takamichitoda/dpc-starter-train)。

# %%
import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm


# ==========================================
# 前處理：統一輸入中的破損標記
# ==========================================
def preprocess_transliteration(text):
    """統一各種破損標記為 <gap> 和 <big_gap>"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\[\.\.\.\]', '<big_gap>', text)      # [...] 先處理
    text = re.sub(r'\.{3,}', '<big_gap>', text)
    text = re.sub(r'…+', '<big_gap>', text)
    text = re.sub(r'xx+', '<gap>', text)
    text = re.sub(r'(?<=\s)x(?=\s)', '<gap>', text)
    text = re.sub(r'<gap>\s*<gap>', '<big_gap>', text)
    text = re.sub(r'<big_gap>\s*<big_gap>', '<big_gap>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ==========================================
# 後處理：清理模型輸出
# ==========================================
SUBSCRIPT_TRANS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SPECIAL_CHAR_TRANS = str.maketrans("ḫḪ", "hH")
FORBIDDEN_CHARS = '!?()"—–<>⌈⌋⌊[]+ʾ/;'
FORBIDDEN_TRANS = str.maketrans('', '', FORBIDDEN_CHARS)

def postprocess_translation(text):
    """清理模型翻譯輸出，正規化 gap、移除不需要的字元"""
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text

    # 1. 特殊字元正規化
    t = t.translate(SPECIAL_CHAR_TRANS)   # ḫ→h, Ḫ→H
    t = t.translate(SUBSCRIPT_TRANS)       # ₀→0, ₁→1, ...

    # 2. 輸出中的 gap 正規化
    t = re.sub(r'(\[x\]|\(x\)|\bx\b)', '<gap>', t, flags=re.I)
    t = re.sub(r'(\.{3,}|…|\[\.+\])', '<big_gap>', t)
    t = re.sub(r'<gap>\s*<gap>', '<big_gap>', t)
    t = re.sub(r'<big_gap>\s*<big_gap>', '<big_gap>', t)

    # 3. 移除註解 (fem), (plur), (?) 等
    t = re.sub(r'\((fem|plur|pl|sing|singular|plural|\?|!)\.?\s*\w*\)', '', t, flags=re.I)

    # 4. 保護 gap token，移除禁止字元，再還原
    t = t.replace('<gap>', '\x00GAP\x00').replace('<big_gap>', '\x00BIG\x00')
    t = t.translate(FORBIDDEN_TRANS)
    t = t.replace('\x00GAP\x00', ' <gap> ').replace('\x00BIG\x00', ' <big_gap> ')

    # 5. 分數正規化
    t = re.sub(r'(\d+)\.5\b', r'\1 ½', t)
    t = re.sub(r'\b0\.5\b', '½', t)
    t = re.sub(r'(\d+)\.25\b', r'\1 ¼', t)
    t = re.sub(r'\b0\.25\b', '¼', t)
    t = re.sub(r'(\d+)\.75\b', r'\1 ¾', t)
    t = re.sub(r'\b0\.75\b', '¾', t)

    # 6. 移除重複詞
    t = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', t)

    # 7. 最終清理
    t = re.sub(r'\s+([.,:])' , r'\1', t)    # 標點前多餘空格
    t = re.sub(r'([.,])\1+', r'\1', t)      # 重複標點
    t = re.sub(r'\s+', ' ', t).strip().strip('-').strip()
    return t

# %%
MODEL_PATH = "/kaggle/input/notebooks/kevindic0214/deepast-starter-train/byt5-base-akkadian"

# %%
TEST_DATA_PATH = "/kaggle/input/competitions/deep-past-initiative-machine-translation/test.csv"
BATCH_SIZE = 16
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 模型載入 ---
print(f"正在從 {MODEL_PATH} 載入模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"=== Model Info: {total_params/1e6:.1f}M params, device={DEVICE} ===")

# --- 資料準備 ---
test_df = pd.read_csv(TEST_DATA_PATH)

# %%
PREFIX = "translate Akkadian to English: "

# 前處理診斷
raw_texts = test_df['transliteration'].astype(str).tolist()
processed_texts = [preprocess_transliteration(t) for t in raw_texts]
pre_changed = sum(1 for a, b in zip(raw_texts, processed_texts) if a != b)
print(f"=== Preprocessing: {pre_changed}/{len(raw_texts)} rows changed by gap normalization ===")
if pre_changed > 0:
    for a, b in zip(raw_texts, processed_texts):
        if a != b:
            print(f"  BEFORE: {a[:80]}")
            print(f"  AFTER:  {b[:80]}")
            print()
            break  # 只顯示 1 筆

class InferenceDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = [preprocess_transliteration(t) for t in df['transliteration']]
        self.texts = [PREFIX + i for i in self.texts]
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text, 
            max_length=MAX_LENGTH, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }

test_dataset = InferenceDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 推論迴圈 ---
print("開始推論...")
all_predictions = []

# %%
with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
  
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True
        )
        
        # ByT5 偶爾會產生超出有效範圍的 token ID
        outputs_np = outputs.cpu().numpy()
        outputs_np = np.where((outputs_np < 0) | (outputs_np >= tokenizer.vocab_size), tokenizer.pad_token_id, outputs_np)
        decoded = tokenizer.batch_decode(outputs_np, skip_special_tokens=True)
        all_predictions.extend([d.strip() for d in decoded])

# %%
# --- 提交檔案 ---
submission = pd.DataFrame({
    "id": test_df["id"],
    "translation": all_predictions
})

# 後處理：清理模型輸出
before_post = submission["translation"].copy()
submission["translation"] = submission["translation"].apply(postprocess_translation)
submission["translation"] = submission["translation"].apply(lambda x: x if len(x) > 0 else "damaged text")

# 後處理診斷
post_changed = (before_post != submission["translation"]).sum()
print(f"=== Postprocessing: {post_changed}/{len(submission)} rows changed ===")
if post_changed > 0:
    changed = before_post[before_post != submission["translation"]]
    for i in changed.index[:3]:
        print(f"  BEFORE: {before_post[i][:80]}")
        print(f"  AFTER:  {submission['translation'][i][:80]}")
        print()

# === 輸出品質統計 ===
lengths = submission["translation"].str.len()
empty_count = (submission["translation"] == "damaged text").sum()
gap_count = submission["translation"].str.count("<gap>").sum()
biggap_count = submission["translation"].str.count("<big_gap>").sum()
print(f"=== Output Statistics ===")
print(f"  Total: {len(submission)} translations")
print(f"  Empty/fallback: {empty_count} rows")
print(f"  Length: min={lengths.min()}, max={lengths.max()}, avg={lengths.mean():.1f}")
print(f"  Gap tokens: {int(gap_count)} <gap>, {int(biggap_count)} <big_gap>")
if empty_count > len(submission) * 0.05:
    print(f"  ⚠️ WARNING: >5% empty translations! Model may be undertrained.")

submission.to_csv("submission.csv", index=False)
print("提交檔案已成功儲存！")
submission.head()

# %%



