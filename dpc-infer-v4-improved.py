# %% [markdown]
# # Deep Past Initiative – 推論 v4：改進版
#
# v4 推論改進（基於 v2-gap infer）：
# - num_beams=8（v2 是 4，更多候選提升品質）
# - length_penalty=1.3（鼓勵更完整的翻譯）
# - repetition_penalty=1.2（減少重複詞/片語）
# - 可選 MBR decoding（從多候選中選 chrF++ 最高者）
#
# MBR decoding 說明：
# 1. 用 beam search 產生 top-K 候選翻譯
# 2. 再用 sampling 產生額外候選
# 3. 計算每個候選與其他候選的平均 chrF++ 相似度
# 4. 選擇相似度最高的候選作為最終翻譯
#

# %%
import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
from collections import Counter


# ==========================================
# 設定
# ==========================================
# 模型路徑（根據 Kaggle 上的實際路徑修改）
MODEL_PATH = "/kaggle/input/notebooks/kevindic0214/dpc-train-v4-tuned/byt5-base-akkadian-v4"

TEST_DATA_PATH = "/kaggle/input/competitions/deep-past-initiative-machine-translation/test.csv"
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 推論參數 ===
NUM_BEAMS = 8               # v2 是 4，增加到 8
LENGTH_PENALTY = 1.3        # >1 鼓勵更長的翻譯
REPETITION_PENALTY = 1.2    # 減少重複

# === MBR 設定 ===
USE_MBR = True              # 設為 False 使用標準 beam search
MBR_NUM_BEAM_CANDS = 4      # beam search 產生的候選數
MBR_NUM_SAMPLE_CANDS = 2    # sampling 產生的額外候選數
MBR_TOP_P = 0.9             # sampling 的 nucleus sampling 參數
MBR_TEMPERATURE = 0.7       # sampling 的溫度

# batch size 根據是否使用 MBR 調整
BATCH_SIZE = 2 if USE_MBR else 8


# ==========================================
# 前處理：統一輸入中的破損標記
# ==========================================
def preprocess_transliteration(text):
    """統一各種破損標記為 <gap> 和 <big_gap>"""
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
    t = t.translate(SPECIAL_CHAR_TRANS)
    t = t.translate(SUBSCRIPT_TRANS)

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
    t = re.sub(r'\s+([.,:])' , r'\1', t)
    t = re.sub(r'([.,])\1+', r'\1', t)
    t = re.sub(r'\s+', ' ', t).strip().strip('-').strip()
    return t


# ==========================================
# MBR decoding（自帶 chrF++ 實作，無需 sacrebleu）
# ==========================================
def _chrfpp_score(hypothesis, reference, max_char_n=6, max_word_n=2, beta=2.0):
    """
    計算 chrF++ 分數（字元 n-gram + 詞 n-gram F-score）。
    無外部依賴，用於 MBR 候選排序。
    """
    if not hypothesis or not reference:
        return 0.0

    total_prec, total_rec, count = 0.0, 0.0, 0

    # 字元 n-gram（n=1..6）
    for n in range(1, max_char_n + 1):
        hyp_ng = Counter(hypothesis[i:i+n] for i in range(len(hypothesis) - n + 1))
        ref_ng = Counter(reference[i:i+n] for i in range(len(reference) - n + 1))
        if not hyp_ng or not ref_ng:
            continue
        common = sum((hyp_ng & ref_ng).values())
        total_prec += common / sum(hyp_ng.values())
        total_rec += common / sum(ref_ng.values())
        count += 1

    # 詞 n-gram（n=1..2，這就是 chrF++ 的 ++ 部分）
    hyp_words = hypothesis.split()
    ref_words = reference.split()
    for n in range(1, max_word_n + 1):
        hyp_ng = Counter(tuple(hyp_words[i:i+n]) for i in range(len(hyp_words) - n + 1))
        ref_ng = Counter(tuple(ref_words[i:i+n]) for i in range(len(ref_words) - n + 1))
        if not hyp_ng or not ref_ng:
            continue
        common = sum((hyp_ng & ref_ng).values())
        total_prec += common / sum(hyp_ng.values())
        total_rec += common / sum(ref_ng.values())
        count += 1

    if count == 0:
        return 0.0
    avg_prec = total_prec / count
    avg_rec = total_rec / count
    if avg_prec + avg_rec == 0:
        return 0.0
    return (1 + beta**2) * avg_prec * avg_rec / (beta**2 * avg_prec + avg_rec)


def mbr_select(candidates):
    """
    從多個候選翻譯中選出與其他候選平均 chrF++ 最高的那個。
    這是 Minimum Bayes Risk (MBR) decoding 的核心思想。
    """
    # 去重但保持順序
    seen = set()
    unique = []
    for c in candidates:
        c = str(c).strip()
        if c and c not in seen:
            unique.append(c)
            seen.add(c)

    if len(unique) == 0:
        return ""
    if len(unique) == 1:
        return unique[0]

    # 計算每個候選與其他候選的平均 chrF++ 相似度
    best_idx, best_score = 0, -1.0
    for i in range(len(unique)):
        total = 0.0
        for j in range(len(unique)):
            if i == j:
                continue
            total += _chrfpp_score(unique[i], unique[j])
        avg = total / (len(unique) - 1)
        if avg > best_score:
            best_score = avg
            best_idx = i

    return unique[best_idx]


# %%
# --- 模型載入 ---
print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"=== Model Info: {total_params/1e6:.1f}M params, device={DEVICE} ===")
print(f"=== Inference Config: beams={NUM_BEAMS}, length_penalty={LENGTH_PENALTY}, "
      f"repetition_penalty={REPETITION_PENALTY}, MBR={USE_MBR} ===")

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
            break

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
print(f"Starting inference... ({len(test_loader)} batches, batch_size={BATCH_SIZE})")
all_predictions = []

# %%
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Translating"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        B = input_ids.shape[0]

        if USE_MBR:
            # === MBR decoding ===
            # Step 1: beam search 候選
            beam_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_LENGTH,
                num_beams=max(NUM_BEAMS, MBR_NUM_BEAM_CANDS),
                num_return_sequences=MBR_NUM_BEAM_CANDS,
                length_penalty=LENGTH_PENALTY,
                repetition_penalty=REPETITION_PENALTY,
                early_stopping=True,
            )
            beam_np = beam_outputs.cpu().numpy()
            beam_np = np.where((beam_np < 0) | (beam_np >= tokenizer.vocab_size), tokenizer.pad_token_id, beam_np)
            beam_texts = tokenizer.batch_decode(beam_np, skip_special_tokens=True)

            # 按 example 分組
            pools = [[] for _ in range(B)]
            for i in range(B):
                pools[i].extend(beam_texts[i * MBR_NUM_BEAM_CANDS:(i + 1) * MBR_NUM_BEAM_CANDS])

            # Step 2: sampling 候選（可選）
            if MBR_NUM_SAMPLE_CANDS > 0:
                sample_outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=MAX_LENGTH,
                    do_sample=True,
                    num_beams=1,
                    top_p=MBR_TOP_P,
                    temperature=MBR_TEMPERATURE,
                    num_return_sequences=MBR_NUM_SAMPLE_CANDS,
                    repetition_penalty=REPETITION_PENALTY,
                )
                samp_np = sample_outputs.cpu().numpy()
                samp_np = np.where((samp_np < 0) | (samp_np >= tokenizer.vocab_size), tokenizer.pad_token_id, samp_np)
                samp_texts = tokenizer.batch_decode(samp_np, skip_special_tokens=True)

                for i in range(B):
                    pools[i].extend(samp_texts[i * MBR_NUM_SAMPLE_CANDS:(i + 1) * MBR_NUM_SAMPLE_CANDS])

            # Step 3: MBR 選擇
            for i in range(B):
                best = mbr_select(pools[i])
                all_predictions.append(best.strip())

        else:
            # === 標準 beam search（v2 + 改進參數）===
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_LENGTH,
                num_beams=NUM_BEAMS,
                length_penalty=LENGTH_PENALTY,
                repetition_penalty=REPETITION_PENALTY,
                early_stopping=True,
            )

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

# 後處理
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

# 輸出品質統計
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
    print(f"  WARNING: >5% empty translations! Model may be undertrained.")

submission.to_csv("submission.csv", index=False)
print("Submission saved!")
submission.head()

# %%


