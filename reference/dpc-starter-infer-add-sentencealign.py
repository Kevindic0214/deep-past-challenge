# %% [markdown]
# # Deep Past Initiative – Machine Translation (Inference Notebook)
# 
# This notebook is a **starter / baseline** for this Kaggle competition.
# 
# Training Code is [here](https://www.kaggle.com/code/takamichitoda/dpc-starter-train).

# %% [markdown]
# # A Rule-Based Baseline Solution (TR-TRY Notebook)
# 
# This notebook builds a **"translation memory bank"** and **"bidirectional confidence dictionary"** using the training set, and generates translation results by matching the most similar training samples for test set texts through multi-dimensional retrieval.
# 
# TR-TRY Notebook is [here](https://www.kaggle.com/code/jackcerion/tr-try). **(copy from @jackcerion)**

# %%
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

# %%
#MODEL_PATH = "/kaggle/input/dpc-starter-train/byt5-akkadian-model/"
#MODEL_PATH = "/kaggle/input/epoch30-seed42/byt5-akkadian-model"
MODEL_PATH="/kaggle/input/k/qifeihhh666/dpc-starter-train/byt5-akkadian-model/"

# %%
TEST_DATA_PATH = "/kaggle/input/deep-past-initiative-machine-translation/test.csv"
BATCH_SIZE = 16
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Loading ---
print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# --- Data Preparation ---
test_df = pd.read_csv(TEST_DATA_PATH)

# %%
PREFIX = "translate Akkadian to English: "

class InferenceDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['transliteration'].astype(str).tolist()
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

# --- Inference Loop ---
print("Starting Inference...")
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
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_predictions.extend([d.strip() for d in decoded])

# %%
# --- Submission ---
submission_Deep = pd.DataFrame({
    "id": test_df["id"],
    "translation": all_predictions
})

submission_Deep["translation"] = submission_Deep["translation"].apply(lambda x: x if len(x) > 0 else "broken text")

submission_Deep.to_csv("submission.csv", index=False)
submission_Deep.head()

# %% [markdown]
# # TR-TRY

# %%
# import re
# import unicodedata
# import numpy as np
# import pandas as pd
# from collections import Counter, defaultdict
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from difflib import SequenceMatcher


# # --- 1. 强化超参数与配置 ---
# USE_PUBLISHED = True
# BATCH = 256
# TOPK = 60
# RERANK_K = 10
# W_CHAR = 0.75   # 初始权重，后续会自动调优
# W_WORD = 0.20
# W_SEQ  = 0.05
# LEN_PENALTY_POWER = 1.5
# MIN_ACCEPT_SCORE = 0.14
# MIN_NNZ = 4


# # 英语停用词过滤（防止词典翻译全是 "the", "and"）
# STOP_WORDS_EN = {"the", "a", "an", "and", "or", "but", "if", "then", "of", "at", "by", "from", "with"}

# # --- 2. 强化的预处理 ---
# def advanced_norm_src(x: str) -> str:
#     if pd.isna(x): return ""
#     # 转换为小写并处理基本空白
#     x = str(x).lower().strip()
#     # 移除阿卡德语下标数字 (u2 -> u, a3 -> a) 以处理同音词折叠
#     x = re.sub(r'[₀-₉0-9]', '', x) 
#     # 处理限定词
#     x = re.sub(r"\{([^}]+)\}", r" DET_\1 ", x)
#     # 移除编辑符号但保留连字符作为音节连接（可选：也可以尝试去掉连字符）
#     x = x.replace("[", "").replace("]", "").replace("<", "").replace(">", "")
#     # 规范化 Unicode (处理 š, ṣ 等)
#     x = unicodedata.normalize("NFKD", x)
#     x = "".join(ch for ch in x if not unicodedata.combining(ch))
#     # 移除特殊撇号
#     x = re.sub(r"[ʾʿˀˁ']", "", x)
#     # 统一 GAP 标记
#     x = re.sub(r"\.{2,}|…|[xX]{1,3}", " GAP ", x)
#     # 最终清理多余空格
#     x = re.sub(r"\s+", " ", x).strip()

#      # determinatives: {d} {m} {ki} ... => DET_d etc
#     x = re.sub(r"\{([^}]+)\}", r" DET_\1 ", x)

#     x = re.sub(r"\(([a-z]{1,6})\)", r" DET_\1 ", x)

#     # remove apostrophe-like signs
#     x = re.sub(r"[ʾʿˀˁ']", "", x)

#     # gap markers
#     x = x.replace("…", " GAP ").replace("...", " GAP ")
#     x = re.sub(r"\b[xX]{1,3}\b", " GAP ", x)

#     # editorial brackets
#     x = x.replace("[", " ").replace("]", " ").replace("<", " ").replace(">", " ")
#     x = x.replace("<<", " ").replace(">>", " ")

#     # dot separators (e.g. KÙ.AN)
#     x = x.replace(".", " ")

#     # plus sometimes appears in sign writing
#     x = x.replace("+", "")

#     # keep ascii 
#     x = re.sub(r"[^a-z0-9\-_ ]+", " ", x)
#     x = re.sub(r"\s+", " ", x).strip()
#     return x

# # --- 3. 构建强化的翻译记忆与双向词典 ---
# train = pd.read_csv('/kaggle/input/deep-past-initiative-machine-translation/train.csv')
# train=train.iloc[1000:]  
# # 模拟切分验证集进行权重调优
# train_df, val_df = train_test_split(train, test_size=0.1, random_state=42)

# def build_memory_and_dict(df):
#     mem = df.copy()
#     mem["src_norm"] = mem["transliteration"].map(advanced_norm_src)
#     mem["tgt_norm"] = mem["translation"].map(lambda x: str(x).lower().strip())
    
#     # 构建双向计数器 (Src -> Tgt 和 Tgt -> Src)
#     src_cnt, tgt_cnt, pair_cnt = Counter(), Counter(), Counter()
    
#     for _, row in mem.iterrows():
#         s_tokens = set(row["src_norm"].split())
#         t_tokens = set([w for w in row["tgt_norm"].split() if w not in STOP_WORDS_EN])
        
#         for sw in s_tokens: src_cnt[sw] += 1
#         for tw in t_tokens: tgt_cnt[tw] += 1
#         for sw in s_tokens:
#             for tw in t_tokens:
#                 pair_cnt[(sw, tw)] += 1
                
#     # 计算具有“双向验证”属性的置信度得分 (Dice Coefficient 变体)
#     best_word = {}
#     for (sw, tw), c in pair_cnt.items():
#         # 分母结合了双方的频率，体现双向置信度
#         score = 2 * c / (src_cnt[sw] + tgt_cnt[tw])
#         if sw not in best_word or score > best_word[sw][1]:
#             best_word[sw] = (tw, score)
            
#     return mem, best_word

# mem_df, prob_dict = build_memory_and_dict(train_df)
# print(f"Memory Built. Dictionary Size: {len(prob_dict)}")


# # --- 4. 混合检索准备 (TF-IDF) ---
# char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6))
# word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))

# # 拟合记忆库
# mem_srcs = mem_df["src_norm"].tolist()
# Xc_mem = char_vec.fit_transform(mem_srcs)
# Xw_mem = word_vec.fit_transform(mem_srcs)
# mem_src_len = np.array([len(s.split()) for s in mem_srcs])

# # --- 5. 核心翻译函数：带重排与回退逻辑 ---
# def length_penalty(q_len, cand_lens):
#     r = cand_lens.astype(np.float32) / max(1, q_len)
#     return np.exp(-np.abs(np.log(r + 1e-5)) * LEN_PENALTY_POWER)

# def translate_query(q, w_c, w_w, w_s):
#     q_norm = advanced_norm_src(q)
#     if not q_norm: return "..."
    
#     # 1. TF-IDF 初筛
#     q_c_vec = char_vec.transform([q_norm])
#     q_w_vec = word_vec.transform([q_norm])
    
#     sc = (q_c_vec @ Xc_mem.T).toarray()[0]
#     sw = (q_w_vec @ Xw_mem.T).toarray()[0]
    
#     combined_scores = w_c * sc + w_w * sw
    
#     # 2. 取 TOPK 并应用长度惩罚
#     cand_indices = np.argpartition(-combined_scores, TOPK)[:TOPK]
#     lps = length_penalty(len(q_norm.split()), mem_src_len[cand_indices])
#     final_search_scores = combined_scores[cand_indices] * lps
    
#     # 3. 精排 (SequenceMatcher)
#     best_score = -1
#     best_idx = -1
    
#     # 在前 RERANK_K 中寻找
#     top_k_indices = cand_indices[np.argsort(-final_search_scores)[:RERANK_K]]
    
#     for idx in top_k_indices:
#         seq_score = SequenceMatcher(None, q_norm, mem_srcs[idx]).ratio()
#         total_s = final_search_scores[np.where(cand_indices==idx)[0][0]] + w_s * seq_score
        
#         if total_s > best_score:
#             best_score = total_s
#             best_idx = idx
            
#     # 4. 判定与回退
#     if best_score > MIN_ACCEPT_SCORE:
#         return mem_df.iloc[best_idx]["translation"]
#     else:
#         # 执行词典回退 (跳过停用词)
#         words = q_norm.split()
#         res = []
#         for w in words:
#             if w in prob_dict and prob_dict[w][1] > 0.1:
#                 res.append(prob_dict[w][0])
#             else:
#                 res.append(w) # 无法翻译则保留原词
#         return " ".join(res)

# # --- 6. 自动化权重调优 (简单网格搜索演示) ---
# print("Optimizing weights on validation set...")
# best_w = (W_CHAR, W_WORD, W_SEQ)
# # 这里可以写一个循环遍历不同的 w_c, w_w 组合，计算验证集的 BLEU

# # --- 7. 执行测试集预测 ---
# test = pd.read_csv('/kaggle/input/deep-past-initiative-machine-translation/test.csv')
# #test=test.head(1000)
# final_preds = []
# for q in test["transliteration"]:
#     final_preds.append(translate_query(q, W_CHAR, W_WORD, W_SEQ))

# # 保存提交
# submission_tr = pd.DataFrame({"id": test["id"], "translation": final_preds})
# submission_tr.to_csv("submission_tr.csv", index=False)
# submission_tr.head()


# %% [markdown]
# # SentenceAlign: append longer segment to shorter string

# %%
# end_result=[]
# for k in range(len(submission_Deep)):
#     text_Deep=submission_Deep.iloc[k]["translation"]
#     text_tr=submission_tr.iloc[k]["translation"]

#     if k < 4:
#         print(text_Deep)
        
#     text_source=test.iloc[k]["transliteration"]
#     if len(text_source)>=256:
#         if len(text_tr)>len(text_Deep):
#             blank_matches=re.finditer(r' ', text_tr)
#             blank_positions = [m.start() for m in blank_matches]
#             if blank_positions:  
#                 text_Deep=text_Deep[:350]
#                 text_Deep_len=len(text_Deep)
#                 closest_blank = min(blank_positions, key=lambda x: abs(text_Deep_len - x))
#                 text_Deep+=text_tr[closest_blank:]
    
#     if not text_Deep.endswith("."):
#         text_Deep += "."

#     if k < 4:
#         print()
#         print(text_Deep)
#         print("*"*100)
        
#     end_result.append(text_Deep)
    
# submission_Deep["translation"]=end_result
# submission_Deep.to_csv("submission.csv",index=False)
# submission_Deep.head()

# %%


# %%



