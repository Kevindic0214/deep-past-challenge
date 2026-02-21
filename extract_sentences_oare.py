"""
從 Sentences_Oare + published_texts 提取句子級翻譯對
產出 sentences_oare_pairs.csv，可直接用於訓練
"""

import pandas as pd
import re
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

BASE = 'D:/Kevin/deep_past_challenge/deep-past-initiative-machine-translation/'
OUTPUT = 'D:/Kevin/deep_past_challenge/sentences_oare_pairs.csv'

# ============================================================
# 1. 載入資料
# ============================================================
print("=== 載入資料 ===")
sent = pd.read_csv(BASE + 'Sentences_Oare_FirstWord_LinNum.csv')
pub = pd.read_csv(BASE + 'published_texts.csv')
train = pd.read_csv(BASE + 'train.csv')

print(f"  Sentences_Oare: {len(sent)} rows")
print(f"  published_texts: {len(pub)} rows")
print(f"  train.csv: {len(train)} rows")

# 有翻譯的句子
sent = sent[sent['translation'].notna()].copy()
print(f"  有翻譯的句子: {len(sent)}")

# 有轉寫的 published_texts
pub = pub[pub['transliteration'].notna()].copy()
print(f"  有轉寫的 published_texts: {len(pub)}")

# 找到交集
overlap_ids = set(sent['text_uuid'].unique()) & set(pub['oare_id'].unique())
print(f"  可配對的文本數: {len(overlap_ids)}")

# 排除已在 train.csv 中的文本（避免資料洩漏）
train_ids = set(train['oare_id'].unique())
new_ids = overlap_ids - train_ids
print(f"  排除 train.csv 後的新文本: {len(new_ids)}")
# 也保留 train 中的文本用於句子級拆分（不算洩漏，只是更細的粒度）
all_ids = overlap_ids
print(f"  包含 train 文本的總數: {len(all_ids)}")

# ============================================================
# 2. 對齊算法
# ============================================================
def align_sentences(transliteration, sentence_rows):
    """
    將句子翻譯對齊到轉寫文本的對應片段

    Args:
        transliteration: 完整轉寫文本
        sentence_rows: 按 sentence_obj_in_text 排序的句子列表

    Returns:
        list of (src_segment, translation) pairs
    """
    words = transliteration.split()
    if not words:
        return []

    sentences = sentence_rows.to_dict('records')
    if not sentences:
        return []

    # Step 1: 找到每個句子的起始位置
    split_points = []
    last_pos = -1

    for rec in sentences:
        fw = str(rec['first_word_spelling'])
        if pd.isna(rec['first_word_spelling']) or fw == 'nan':
            split_points.append(None)
            continue

        # 在 transliteration 中找 first_word，必須在上一個位置之後
        found = None
        for i, w in enumerate(words):
            if i <= last_pos:
                continue
            # 精確匹配或前綴匹配（處理 OCR 變體）
            if w == fw:
                found = i
                break

        # 如果精確匹配失敗，嘗試忽略下標數字的匹配
        if found is None:
            fw_normalized = re.sub(r'[₀₁₂₃₄₅₆₇₈₉0-9]', '', fw)
            for i, w in enumerate(words):
                if i <= last_pos:
                    continue
                w_normalized = re.sub(r'[₀₁₂₃₄₅₆₇₈₉0-9]', '', w)
                if w_normalized == fw_normalized and fw_normalized:
                    found = i
                    break

        split_points.append(found)
        if found is not None:
            last_pos = found

    # Step 2: 根據 split_points 切分 transliteration
    pairs = []
    for idx, rec in enumerate(sentences):
        start = split_points[idx]

        # 第一個句子：從開頭開始
        if idx == 0:
            start = 0

        if start is None:
            continue

        # 終點：下一個句子的起始，或文本結尾
        end = len(words)
        for j in range(idx + 1, len(sentences)):
            if split_points[j] is not None:
                end = split_points[j]
                break

        segment = ' '.join(words[start:end]).strip()
        translation = str(rec['translation']).strip()

        if segment and translation and translation != 'nan':
            pairs.append({
                'transliteration': segment,
                'translation': translation,
            })

    return pairs


# ============================================================
# 3. 執行對齊
# ============================================================
print("\n=== 執行句子對齊 ===")

all_pairs = []
stats = {
    'total_texts': 0,
    'success_texts': 0,
    'failed_texts': 0,
    'total_pairs': 0,
    'from_new_texts': 0,
    'from_train_texts': 0,
}

for tid in all_ids:
    stats['total_texts'] += 1

    # 取得轉寫
    pub_row = pub[pub['oare_id'] == tid]
    if len(pub_row) == 0:
        continue
    translit = str(pub_row.iloc[0]['transliteration'])

    # 取得該文本的所有句子，按順序排列
    text_sents = sent[sent['text_uuid'] == tid].sort_values('sentence_obj_in_text')

    # 對齊
    pairs = align_sentences(translit, text_sents)

    if pairs:
        stats['success_texts'] += 1
        stats['total_pairs'] += len(pairs)
        if tid in train_ids:
            stats['from_train_texts'] += len(pairs)
        else:
            stats['from_new_texts'] += len(pairs)

        for p in pairs:
            p['oare_id'] = tid
            p['source'] = 'train_split' if tid in train_ids else 'new'
        all_pairs.extend(pairs)
    else:
        stats['failed_texts'] += 1

print(f"  處理文本數: {stats['total_texts']}")
print(f"  成功對齊: {stats['success_texts']}")
print(f"  失敗: {stats['failed_texts']}")
print(f"  總配對數: {stats['total_pairs']}")
print(f"    來自新文本: {stats['from_new_texts']}")
print(f"    來自 train 文本（句子級拆分）: {stats['from_train_texts']}")

# ============================================================
# 4. 品質檢查
# ============================================================
print("\n=== 品質檢查 ===")
df_pairs = pd.DataFrame(all_pairs)

# 基本統計
print(f"  總配對數: {len(df_pairs)}")
print(f"  平均轉寫長度: {df_pairs['transliteration'].str.len().mean():.1f} chars")
print(f"  平均翻譯長度: {df_pairs['translation'].str.len().mean():.1f} chars")

# 過濾太短或太長的配對
min_src_len = 5
max_src_len = 2000
min_tgt_len = 3
max_tgt_len = 2000

before = len(df_pairs)
df_pairs = df_pairs[
    (df_pairs['transliteration'].str.len() >= min_src_len) &
    (df_pairs['transliteration'].str.len() <= max_src_len) &
    (df_pairs['translation'].str.len() >= min_tgt_len) &
    (df_pairs['translation'].str.len() <= max_tgt_len)
]
print(f"  過濾後: {len(df_pairs)} (移除 {before - len(df_pairs)} 筆)")

# 過濾非英文翻譯（有些可能是德文、法文等）
def is_english(text):
    """簡單判斷是否為英文翻譯"""
    # 常見非英文標記
    non_english_words = ['der', 'die', 'das', 'und', 'von', 'les', 'des', 'une', 'est',
                         'Vom', 'Ein', 'für', 'dem']
    words = text.split()[:10]  # 只看前10個詞
    non_en_count = sum(1 for w in words if w in non_english_words)
    # 如果前10個詞中有3個以上非英文詞，判定為非英文
    return non_en_count < 3

before = len(df_pairs)
df_pairs = df_pairs[df_pairs['translation'].apply(is_english)]
print(f"  過濾非英文後: {len(df_pairs)} (移除 {before - len(df_pairs)} 筆)")

# 移除重複
before = len(df_pairs)
df_pairs = df_pairs.drop_duplicates(subset=['transliteration', 'translation'])
print(f"  去重後: {len(df_pairs)} (移除 {before - len(df_pairs)} 筆)")

# ============================================================
# 5. 顯示範例
# ============================================================
print("\n=== 範例配對 ===")
new_pairs = df_pairs[df_pairs['source'] == 'new']
for i in range(min(5, len(new_pairs))):
    row = new_pairs.iloc[i]
    print(f"\n--- Pair {i+1} ---")
    print(f"  Src: {row['transliteration'][:200]}")
    print(f"  Tgt: {row['translation'][:200]}")

# ============================================================
# 6. 儲存
# ============================================================
# 只保存需要的欄位
output_df = df_pairs[['oare_id', 'transliteration', 'translation', 'source']].copy()
output_df.to_csv(OUTPUT, index=False, encoding='utf-8')
print(f"\n=== 已儲存到 {OUTPUT} ===")
print(f"  總計: {len(output_df)} 筆")
print(f"  新資料: {len(output_df[output_df['source'] == 'new'])} 筆")
print(f"  train 拆分: {len(output_df[output_df['source'] == 'train_split'])} 筆")
