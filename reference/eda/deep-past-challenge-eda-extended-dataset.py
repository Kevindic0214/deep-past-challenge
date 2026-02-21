# %% [markdown]
# # 🏆 Deep Past Challenge - EDA + Extended Dataset
# ## Translating 4,000-Year-Old Akkadian Cuneiform Tablets
# 
# ---
# 
# ### ⭐ If this notebook helps you, please upvote! ⭐
# 
# ---
# 
# ## 📜 Competition Overview
# 
# | Item | Details |
# |------|--------|
# | 💰 **Prize Pool** | $50,000 |
# | 🌍 **Language** | Old Assyrian (Akkadian dialect), ~1950-1850 BC |
# | ✍️ **Script** | Cuneiform - world's first writing system |
# | 📝 **Content** | Commercial records, contracts, letters |
# | 📊 **Challenge** | ~22,000 tablets, ~50% still untranslated |
# 
# ## 📚 What You'll Learn
# 
# 1. Dataset structure and statistics
# 2. 💡 **Critical Discovery**: Test data structure reveals key insight
# 3. Character and vocabulary analysis
# 4. Lexicon coverage
# 5. Supplementary data opportunities
# 6. Recommended approaches
# 7. 📊 **Extended Dataset**: 7,953 texts + enhanced dictionary
# 
# ---

# %% [markdown]
# ## 1. 📦 Setup and Data Loading

# %%
# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

print("Libraries loaded!")

# %%
# Path detection (Kaggle vs local)
DATA_PATH = '/kaggle/input/deep-past-initiative-machine-translation/' if os.path.exists('/kaggle/input') else './data/'

# Main datasets
train = pd.read_csv(f'{DATA_PATH}train.csv')
test = pd.read_csv(f'{DATA_PATH}test.csv')
sample_submission = pd.read_csv(f'{DATA_PATH}sample_submission.csv')

# Supplementary datasets
lexicon = pd.read_csv(f'{DATA_PATH}OA_Lexicon_eBL.csv')
published_texts = pd.read_csv(f'{DATA_PATH}published_texts.csv')

print("Dataset Sizes:")
print(f"  train.csv:        {len(train):>7,} parallel examples")
print(f"  test.csv:         {len(test):>7,} samples to translate")
print(f"  lexicon:          {len(lexicon):>7,} dictionary entries")
print(f"  published_texts:  {len(published_texts):>7,} additional texts")

# %% [markdown]
# ## 2. 💡 Critical Discovery: Test Data Structure
# 
# Before diving into general EDA, let's examine the test data structure - this reveals a **crucial insight** for this competition.

# %%
print("Test Data Structure Analysis")
print("=" * 60)
print(f"\nNumber of test samples: {len(test)}")
print(f"Unique text_id values: {test['text_id'].nunique()}")
print(f"\nText ID: {test['text_id'].iloc[0]}")
print("\nSegment Details:")
print("-" * 60)

for _, row in test.iterrows():
    print(f"  ID {row['id']}: Lines {row['line_start']:2d}-{row['line_end']:2d} | {len(row['transliteration']):3d} characters")

# %%
# Visualize test data structure
fig, ax = plt.subplots(figsize=(12, 4))

colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
for idx, row in test.iterrows():
    ax.barh(0, row['line_end'] - row['line_start'] + 1, 
            left=row['line_start'] - 1, height=0.5, 
            color=colors[idx], alpha=0.8, edgecolor='white', linewidth=2,
            label=f"ID {row['id']}: Lines {row['line_start']}-{row['line_end']}")
    ax.text((row['line_start'] + row['line_end']) / 2 - 0.5, 0, 
            f"ID {row['id']}", ha='center', va='center', fontsize=12, fontweight='bold', color='white')

ax.set_xlim(0, test['line_end'].max() + 1)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel('Line Number', fontsize=12)
ax.set_yticks([])
ax.set_title('Test Data: 4 Consecutive Segments from ONE Ancient Text', fontsize=14, fontweight='bold')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
plt.tight_layout()
plt.show()

print("\nKEY INSIGHT: All 4 test samples are consecutive segments of the SAME ancient document!")
print("This means we can potentially find ONE similar training example and segment it proportionally.")

# %% [markdown]
# ## 3. 🎯 Similarity Analysis: Finding Matching Training Samples
# 
# Let's check if there's a training sample similar to our test text.
# 
# > 📚 **Research Note**: [Berkeley's CuneiTranslate](https://www.ischool.berkeley.edu/projects/2024/cuneitranslate-unlocking-ancient-mesopotamian-knowledge) found neural MT (T5, mT5, NLLB) achieved **BLEU < 8%** on cuneiform translation due to limited data. High similarity matches make retrieval-based approaches more effective!

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Combine all test segments into one text
full_test_text = ' '.join(test['transliteration'].tolist())

# Build TF-IDF with character n-grams (works well for morphologically rich languages)
vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(2, 6),
    max_features=25000,
    sublinear_tf=True
)

train_vectors = vectorizer.fit_transform(train['transliteration'].str.lower())
test_vector = vectorizer.transform([full_test_text.lower()])

# Find most similar training samples
similarities = cosine_similarity(test_vector, train_vectors)[0]

print("Top 5 Most Similar Training Samples to Test Text")
print("=" * 60)

top_indices = similarities.argsort()[-5:][::-1]
for rank, idx in enumerate(top_indices, 1):
    sim = similarities[idx]
    preview = train.iloc[idx]['transliteration'][:60]
    print(f"\n{rank}. Similarity: {sim:.1%}")
    print(f"   Preview: {preview}...")

# %%
# Visualize similarity distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(similarities, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(similarities.max(), color='red', linestyle='--', linewidth=2, label=f'Best Match: {similarities.max():.1%}')
axes[0].set_xlabel('Cosine Similarity')
axes[0].set_ylabel('Number of Training Samples')
axes[0].set_title('Similarity Distribution: Test vs All Training Samples')
axes[0].legend()

# Top 20 similarities
top_20_sims = sorted(similarities, reverse=True)[:20]
axes[1].bar(range(1, 21), top_20_sims, color='coral', edgecolor='white')
axes[1].set_xlabel('Rank')
axes[1].set_ylabel('Cosine Similarity')
axes[1].set_title('Top 20 Most Similar Training Samples')
axes[1].set_xticks(range(1, 21))

plt.tight_layout()
plt.show()

print(f"\nBest match similarity: {similarities.max():.1%}")
print(f"This high similarity suggests retrieval-based translation could work well!")

# %% [markdown]
# ## 4. 📊 Training Data Analysis

# %%
print("Training Data Overview")
print("=" * 60)
print(f"Shape: {train.shape}")
print(f"Columns: {train.columns.tolist()}")
print(f"\nMissing values: {train.isnull().sum().sum()}")

# %%
# Sample training examples
print("Sample Training Example")
print("=" * 60)
print("\nTransliteration (Akkadian):")
print(train.iloc[0]['transliteration'][:400] + "...")
print("\nTranslation (English):")
print(train.iloc[0]['translation'][:400] + "...")

# %%
# Text length analysis
train['trans_len'] = train['transliteration'].str.len()
train['transl_len'] = train['translation'].str.len()
train['trans_words'] = train['transliteration'].str.split().str.len()
train['transl_words'] = train['translation'].str.split().str.len()

print("Text Length Statistics")
print("=" * 60)
print("\nTransliteration (Akkadian):")
print(f"  Characters: min={train['trans_len'].min()}, max={train['trans_len'].max()}, median={train['trans_len'].median():.0f}")
print(f"  Words: min={train['trans_words'].min()}, max={train['trans_words'].max()}, median={train['trans_words'].median():.0f}")
print("\nTranslation (English):")
print(f"  Characters: min={train['transl_len'].min()}, max={train['transl_len'].max()}, median={train['transl_len'].median():.0f}")
print(f"  Words: min={train['transl_words'].min()}, max={train['transl_words'].max()}, median={train['transl_words'].median():.0f}")

# %%
# Visualize text length distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(train['trans_len'], bins=50, alpha=0.8, color='steelblue', edgecolor='white')
axes[0, 0].axvline(train['trans_len'].median(), color='red', linestyle='--', label=f'Median: {train["trans_len"].median():.0f}')
axes[0, 0].set_xlabel('Characters')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Transliteration - Character Length')
axes[0, 0].legend()

axes[0, 1].hist(train['transl_len'], bins=50, alpha=0.8, color='coral', edgecolor='white')
axes[0, 1].axvline(train['transl_len'].median(), color='red', linestyle='--', label=f'Median: {train["transl_len"].median():.0f}')
axes[0, 1].set_xlabel('Characters')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Translation - Character Length')
axes[0, 1].legend()

axes[1, 0].hist(train['trans_words'], bins=50, alpha=0.8, color='steelblue', edgecolor='white')
axes[1, 0].axvline(train['trans_words'].median(), color='red', linestyle='--', label=f'Median: {train["trans_words"].median():.0f}')
axes[1, 0].set_xlabel('Words')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Transliteration - Word Count')
axes[1, 0].legend()

axes[1, 1].hist(train['transl_words'], bins=50, alpha=0.8, color='coral', edgecolor='white')
axes[1, 1].axvline(train['transl_words'].median(), color='red', linestyle='--', label=f'Median: {train["transl_words"].median():.0f}')
axes[1, 1].set_xlabel('Words')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Translation - Word Count')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# %%
# Length correlation analysis
train['length_ratio'] = train['transl_len'] / train['trans_len']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(train['trans_len'], train['transl_len'], alpha=0.5, s=20, c='steelblue')
z = np.polyfit(train['trans_len'], train['transl_len'], 1)
p = np.poly1d(z)
axes[0].plot(train['trans_len'].sort_values(), p(train['trans_len'].sort_values()), "r--", alpha=0.8)
corr = train['trans_len'].corr(train['transl_len'])
axes[0].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[0].transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[0].set_xlabel('Transliteration Length (chars)')
axes[0].set_ylabel('Translation Length (chars)')
axes[0].set_title('Length Correlation')

axes[1].hist(train['length_ratio'], bins=50, alpha=0.8, color='green', edgecolor='white')
axes[1].axvline(train['length_ratio'].median(), color='red', linestyle='--', label=f'Median: {train["length_ratio"].median():.2f}')
axes[1].set_xlabel('Translation/Transliteration Ratio')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Length Ratio Distribution')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Length ratio: mean={train['length_ratio'].mean():.2f}, std={train['length_ratio'].std():.2f}")
print("This ratio is useful for estimating expected translation length.")

# %% [markdown]
# ## 5. 🔤 Character and Vocabulary Analysis

# %%
# Character analysis
all_trans_text = ' '.join(train['transliteration'].values)
trans_chars = Counter(all_trans_text)

# Special characters (non-ASCII)
special_chars = {char: count for char, count in trans_chars.items() if ord(char) > 127}

print("Character Analysis")
print("=" * 60)
print(f"\nTotal unique characters: {len(trans_chars)}")
print(f"Special characters (non-ASCII): {len(special_chars)}")
print("\nTop 20 Special Characters:")
for char, count in sorted(special_chars.items(), key=lambda x: -x[1])[:20]:
    print(f"  '{char}' (U+{ord(char):04X}): {count:,}")

# %%
# Logograms/Sumerograms (ALL CAPS words)
logogram_pattern = r'\b[A-Z][A-Z0-9]+\b'
logograms = []
for text in train['transliteration']:
    logograms.extend(re.findall(logogram_pattern, text))

logogram_counts = Counter(logograms)

print("Logograms/Sumerograms Analysis")
print("=" * 60)
print(f"\nUnique logograms: {len(logogram_counts):,}")
print(f"Total occurrences: {sum(logogram_counts.values()):,}")
print("\nTop 20 Most Common:")
for logo, count in logogram_counts.most_common(20):
    print(f"  {logo}: {count:,}")

# %%
# Vocabulary analysis
trans_words_all = []
for text in train['transliteration']:
    trans_words_all.extend(text.lower().split())

transl_words_all = []
for text in train['translation']:
    transl_words_all.extend(text.lower().split())

trans_vocab = Counter(trans_words_all)
transl_vocab = Counter(transl_words_all)

print("Vocabulary Statistics")
print("=" * 60)
print(f"\nTransliteration vocabulary: {len(trans_vocab):,} unique tokens")
print(f"Translation vocabulary: {len(transl_vocab):,} unique tokens")
print(f"\nTotal transliteration tokens: {len(trans_words_all):,}")
print(f"Total translation tokens: {len(transl_words_all):,}")

# %%
# Top vocabulary
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

top_trans = dict(trans_vocab.most_common(20))
axes[0].barh(list(top_trans.keys())[::-1], list(top_trans.values())[::-1], color='steelblue', edgecolor='white')
axes[0].set_xlabel('Frequency')
axes[0].set_title('Top 20 Transliteration Tokens')

top_transl = dict(transl_vocab.most_common(20))
axes[1].barh(list(top_transl.keys())[::-1], list(top_transl.values())[::-1], color='coral', edgecolor='white')
axes[1].set_xlabel('Frequency')
axes[1].set_title('Top 20 Translation Tokens')

plt.tight_layout()
plt.show()

# %%
# Zipf's Law
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

trans_freq = sorted(trans_vocab.values(), reverse=True)
axes[0].loglog(range(1, len(trans_freq) + 1), trans_freq, 'b-', alpha=0.7)
axes[0].set_xlabel('Rank')
axes[0].set_ylabel('Frequency')
axes[0].set_title("Zipf's Law - Transliteration")
axes[0].grid(True, alpha=0.3)

transl_freq = sorted(transl_vocab.values(), reverse=True)
axes[1].loglog(range(1, len(transl_freq) + 1), transl_freq, 'r-', alpha=0.7)
axes[1].set_xlabel('Rank')
axes[1].set_ylabel('Frequency')
axes[1].set_title("Zipf's Law - Translation")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Rare Word Analysis:")
for threshold in [1, 2, 5]:
    trans_rare = sum(1 for c in trans_vocab.values() if c <= threshold)
    print(f"  Words appearing <= {threshold} times: {trans_rare} ({trans_rare/len(trans_vocab)*100:.1f}%)")

# %% [markdown]
# ## 6. 📖 Lexicon Analysis

# %%
print("Lexicon Overview")
print("=" * 60)
print(f"Shape: {lexicon.shape}")
print(f"Columns: {lexicon.columns.tolist()}")
print(f"\nEntry types:")
for t, c in lexicon['type'].value_counts().items():
    print(f"  {t}: {c:,} ({c/len(lexicon)*100:.1f}%)")

# %%
# Lexicon coverage
lexicon_forms = set(lexicon['form'].dropna().str.lower())
lexicon_norms = set(lexicon['norm'].dropna().str.lower())

covered = sum(1 for w in trans_vocab.keys() if w in lexicon_forms or w in lexicon_norms)
oov = len(trans_vocab) - covered

print("Lexicon Coverage Analysis")
print("=" * 60)
print(f"\nVocabulary covered by lexicon: {covered}/{len(trans_vocab)} ({covered/len(trans_vocab)*100:.1f}%)")
print(f"Out-of-vocabulary words: {oov} ({oov/len(trans_vocab)*100:.1f}%)")

# Visualize
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie([covered, oov], labels=['In Lexicon', 'OOV'], autopct='%1.1f%%', 
       colors=['#2ecc71', '#e74c3c'], startangle=90, explode=(0.05, 0))
ax.set_title('Vocabulary Lexicon Coverage')
plt.show()

# %% [markdown]
# ## 7. 📜 Published Texts Analysis

# %%
print("Published Texts Overview")
print("=" * 60)
print(f"Shape: {published_texts.shape}")
print(f"\nTexts with AICC translation link: {published_texts['AICC_translation'].notna().sum():,}")
print("  (Potential additional training data!)")

# %%
# Genre distribution
genre_counts = published_texts['genre_label'].value_counts()

fig, ax = plt.subplots(figsize=(12, 6))
genre_counts.plot(kind='bar', ax=ax, color='purple', edgecolor='white', alpha=0.8)
ax.set_xlabel('Genre')
ax.set_ylabel('Count')
ax.set_title('Published Texts by Genre')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\nGenre Distribution:")
for genre, count in genre_counts.head(10).items():
    print(f"  {genre}: {count:,} ({count/len(published_texts)*100:.1f}%)")

# %% [markdown]
# ## 8. 🔍 Text Structure Patterns

# %%
# Analyze text structure patterns
patterns = {
    'Numbered lines': r'\d+\'?\.',
    'Tablet sides (obv/rev)': r'\b(obv\.|rev\.|o\.|r\.)\b',
    'Broken text [...]': r'\[.*?\]',
    'Uncertain (...)': r'\(.*?\)',
    'Seal mentions': r'\b(seal|KIŠIB)\b',
}

print("Text Structure Patterns")
print("=" * 60)
for name, pattern in patterns.items():
    count = sum(1 for text in train['transliteration'] if re.search(pattern, text, re.IGNORECASE))
    print(f"  {name}: {count} texts ({count/len(train)*100:.1f}%)")

# Bracket analysis
train['brackets'] = train['transliteration'].apply(lambda x: len(re.findall(r'\[', x)))
print(f"\nBroken text indicators [brackets]: mean={train['brackets'].mean():.1f} per text")

# %% [markdown]
# ## 9. 📈 Key Insights Summary

# %%
# Summary visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Dataset sizes
datasets = ['Train', 'Test', 'Lexicon', 'Published']
sizes = [len(train), len(test), len(lexicon), len(published_texts)]
colors = ['steelblue', 'coral', 'teal', 'purple']
bars = axes[0, 0].bar(datasets, sizes, color=colors, edgecolor='white')
axes[0, 0].set_ylabel('Records')
axes[0, 0].set_title('Dataset Sizes')
axes[0, 0].set_yscale('log')
for bar, s in zip(bars, sizes):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, s, f'{s:,}', ha='center', va='bottom')

# Similarity histogram
axes[0, 1].hist(similarities, bins=30, color='green', edgecolor='white', alpha=0.8)
axes[0, 1].axvline(similarities.max(), color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Similarity')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title(f'Test-Train Similarity (Best: {similarities.max():.1%})')

# Vocabulary coverage
axes[1, 0].pie([covered, oov], labels=['In Lexicon', 'OOV'], autopct='%1.1f%%', 
               colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1, 0].set_title('Lexicon Coverage')

# Length distribution
axes[1, 1].boxplot([train['trans_len'], train['transl_len']], labels=['Akkadian', 'English'])
axes[1, 1].set_ylabel('Characters')
axes[1, 1].set_title('Text Length Distribution')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. 🔮 Recommendations
# 
# Based on this EDA, here are the key findings and recommendations:
# 
# ### 💡 Key Findings
# 
# | Finding | Implication |
# |---------|-------------|
# | 🎯 All 4 test samples from ONE text | Retrieval-based approach works well |
# | 📊 ~87% similarity match exists | High-quality translation retrieval possible |
# | ⚠️ Only 1,561 training samples | Neural MT overfits; retrieval is better |
# | 📖 ~37% lexicon coverage | Dictionary lookup can supplement translation |
# | 📐 Strong length correlation (0.96) | Length ratio helps estimate output size |
# 
# ### 📚 Research Context
# 
# | Paper/Project | Method | Result |
# |---------------|--------|--------|
# | [Berkeley CuneiTranslate](https://www.ischool.berkeley.edu/projects/2024/cuneitranslate-unlocking-ancient-mesopotamian-knowledge) | T5, mT5, NLLB | BLEU < 8% ❌ |
# | [PNAS Nexus 2023](https://academic.oup.com/pnasnexus/article/2/5/pgad096/7147349) | CNN + BPE | BLEU 37.47 ✅ |
# | Our Retrieval Approach | TF-IDF + Segmentation | 87% match ✅ |
# 
# ### 🚀 Recommended Approaches
# 
# 1. **Baseline (No GPU)**
#    - TF-IDF similarity matching with character n-grams
#    - Proportional segmentation for line-based extraction
# 
# 2. **Enhanced**
#    - BM25 + TF-IDF hybrid scoring
#    - Lexicon-augmented post-processing
# 
# 3. **Advanced (GPU required)**
#    - Fine-tune multilingual models (mBART, mT5)
#    - Pre-train on published_texts first
# 
# 4. **Data Augmentation**
#    - Extract translations from AICC links in published_texts
#    - Use publications.csv OCR data
# 
# ---
# 
# ### ⭐ If this notebook helped you, please upvote! ⭐
# 
# 💬 Feel free to ask questions in the comments!

# %% [markdown]
# ## 11. 📊 Competition Data vs Extended Dataset
# 
# We created an **extended dataset** to help competitors. Let's compare:
# 
# | Dataset | Source | Purpose |
# |---------|--------|--------|
# | **Competition Data** | Kaggle | Official train/test |
# | **Extended Dataset** | Our contribution | Additional resources |

# %%
# ============================================
# COMPETITION DATA (Original)
# ============================================
print('='*70)
print('COMPETITION DATA (from Kaggle)')
print('='*70)

print('\n📁 Files provided by competition:')
print(f'  train.csv:           {len(train):>7,} parallel texts')
print(f'  test.csv:            {len(test):>7,} texts to translate')
print(f'  OA_Lexicon_eBL.csv:  {len(lexicon):>7,} dictionary entries')
print(f'  published_texts.csv: {len(published_texts):>7,} text metadata')

print('\n📊 train.csv columns:')
print(f'  {train.columns.tolist()}')
print(f'  → Only {len(train.columns)} columns: ID + source + target')

# %%
# ============================================
# EXTENDED DATASET (Our Contribution)
# ============================================

# Try to load extended dataset
EXT_PATH = '/kaggle/input/old-assyrian-extended-corpus/' if os.path.exists('/kaggle/input/old-assyrian-extended-corpus') else './kaggle-dataset-v2/'

try:
    corpus = pd.read_csv(f'{EXT_PATH}akkadian_corpus.csv')
    dictionary = pd.read_csv(f'{EXT_PATH}akkadian_dictionary.csv')
    EXTENDED_AVAILABLE = True
    
    parallel = corpus[corpus['has_translation'] == True]
    monolingual = corpus[corpus['has_translation'] == False]
    
    print('='*70)
    print('EXTENDED DATASET (Our Contribution)')
    print('='*70)
    
    print('\n📁 Files in extended dataset:')
    print(f'  akkadian_corpus.csv:     {len(corpus):>7,} total texts')
    print(f'    ├─ Parallel:           {len(parallel):>7,} (with translation)')
    print(f'    └─ Monolingual:        {len(monolingual):>7,} (source only)')
    print(f'  akkadian_dictionary.csv: {len(dictionary):>7,} entries')
    
    print('\n📊 akkadian_corpus.csv columns:')
    print(f'  {corpus.columns.tolist()}')
    print(f'  → {len(corpus.columns)} columns with rich metadata!')
    
except FileNotFoundError:
    print('Extended dataset not found. Download from Kaggle:')
    print('  kaggle datasets download -d leiwong/old-assyrian-extended-corpus')
    EXTENDED_AVAILABLE = False

# %% [markdown]
# ### 11.1 Side-by-Side Comparison

# %%
if EXTENDED_AVAILABLE:
    print('='*70)
    print('SIDE-BY-SIDE COMPARISON')
    print('='*70)
    
    comparison = [
        ['Metric', 'Competition Data', 'Extended Dataset', 'Difference'],
        ['─'*15, '─'*20, '─'*20, '─'*15],
        ['Parallel Texts', f'{len(train):,}', f'{len(parallel):,}', 'Same content'],
        ['Monolingual Texts', '0', f'{len(monolingual):,}', f'+{len(monolingual):,} NEW'],
        ['Total Texts', f'{len(train):,}', f'{len(corpus):,}', f'+{len(monolingual):,} (+{len(monolingual)/len(train)*100:.0f}%)'],
        ['Metadata Columns', f'{len(train.columns)}', f'{len(corpus.columns)}', f'+{len(corpus.columns)-len(train.columns)} columns'],
        ['Genre Labels', 'No', 'Yes', 'NEW'],
        ['CDLI IDs', 'No', 'Yes', 'NEW'],
        ['Quality Metrics', 'No', 'Yes (gap_count)', 'NEW'],
        ['Dictionary', f'{len(lexicon):,}', f'{len(dictionary):,}', f'+{len(dictionary)-len(lexicon):,}'],
        ['Word Frequencies', 'No', 'Yes', 'NEW'],
        ['Logogram Meanings', 'No', 'Yes (27)', 'NEW'],
    ]
    
    for row in comparison:
        print(f'{row[0]:<18} {row[1]:<22} {row[2]:<22} {row[3]}')

# %%
if EXTENDED_AVAILABLE:
    # Visual comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Text count comparison
    labels = ['Competition\ntrain.csv', 'Extended\nParallel', 'Extended\nMonolingual']
    sizes = [len(train), len(parallel), len(monolingual)]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = axes[0].bar(labels, sizes, color=colors, edgecolor='white', linewidth=2)
    axes[0].set_ylabel('Number of Texts', fontsize=12)
    axes[0].set_title('Text Count Comparison', fontsize=14, fontweight='bold')
    for bar, size in zip(bars, sizes):
        axes[0].text(bar.get_x() + bar.get_width()/2, size + 150, 
                     f'{size:,}', ha='center', fontsize=12, fontweight='bold')
    
    # 2. Column count comparison
    col_labels = ['Competition\ntrain.csv', 'Extended\ncorpus.csv']
    col_counts = [len(train.columns), len(corpus.columns)]
    bars2 = axes[1].bar(col_labels, col_counts, color=['#3498db', '#2ecc71'], edgecolor='white', linewidth=2)
    axes[1].set_ylabel('Number of Columns', fontsize=12)
    axes[1].set_title('Metadata Richness', fontsize=14, fontweight='bold')
    for bar, cnt in zip(bars2, col_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, cnt + 0.3, 
                     f'{cnt}', ha='center', fontsize=14, fontweight='bold')
    
    # 3. Data composition pie
    axes[2].pie([len(parallel), len(monolingual)], 
                labels=['With Translation\n(Parallel)', 'Without Translation\n(Monolingual)'],
                autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'],
                explode=(0.03, 0.03), startangle=90,
                textprops={'fontsize': 11})
    axes[2].set_title(f'Extended Dataset Composition\n({len(corpus):,} total)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### 11.2 New Columns in Extended Dataset

# %%
if EXTENDED_AVAILABLE:
    # Show what's new
    original_cols = set(train.columns)
    extended_cols = set(corpus.columns)
    new_cols = extended_cols - original_cols
    
    print('NEW COLUMNS in Extended Dataset')
    print('='*70)
    
    for col in sorted(new_cols):
        sample = corpus[col].dropna()
        if len(sample) > 0:
            val = sample.iloc[0]
            if isinstance(val, str) and len(val) > 35:
                val = val[:35] + '...'
            print(f'  {col:25s} Example: {val}')
        else:
            print(f'  {col:25s} (no examples)')

# %% [markdown]
# ### 11.3 Genre Distribution (NEW in Extended)

# %%
if EXTENDED_AVAILABLE:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Parallel texts genre
    parallel_genre = parallel['genre_label'].value_counts().head(8)
    parallel_genre.plot(kind='barh', ax=axes[0], color='#2ecc71', edgecolor='white')
    axes[0].set_xlabel('Count')
    axes[0].set_title('Parallel Texts by Genre (with translation)', fontweight='bold')
    axes[0].invert_yaxis()
    
    # Monolingual texts genre
    mono_genre = monolingual['genre_label'].value_counts().head(8)
    mono_genre.plot(kind='barh', ax=axes[1], color='#e74c3c', edgecolor='white')
    axes[1].set_xlabel('Count')
    axes[1].set_title('Monolingual Texts by Genre (no translation)', fontweight='bold')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    print('\nGenre information helps filter texts for domain-specific models!')

# %% [markdown]
# ### 11.4 Dictionary Enhancements

# %%
if EXTENDED_AVAILABLE:
    print('DICTIONARY COMPARISON')
    print('='*70)
    
    print('\nCompetition Lexicon (OA_Lexicon_eBL.csv):')
    print(f'  Entries: {len(lexicon):,}')
    print(f'  Columns: {lexicon.columns.tolist()}')
    print(f'  Word frequency info: NO')
    
    print('\nExtended Dictionary (akkadian_dictionary.csv):')
    print(f'  Entries: {len(dictionary):,}')
    
    lex_entries = dictionary[dictionary['entry_type'] == 'lexicon']
    logo_entries = dictionary[dictionary['entry_type'] == 'logogram']
    in_train = dictionary[dictionary['in_train_data'] == True]
    with_meaning = dictionary[dictionary['known_meaning'].notna()]
    
    print(f'    ├─ Lexicon entries: {len(lex_entries):,}')
    print(f'    └─ Logogram entries: {len(logo_entries):,}')
    print(f'  Words in training data: {len(in_train):,}')
    print(f'  Logograms with meanings: {len(with_meaning)}')
    
    print('\nTop 10 Logograms with Meanings:')
    print('-'*50)
    for _, row in with_meaning.head(10).iterrows():
        print(f"  {row['form']:15s} = {row['known_meaning']}")

# %% [markdown]
# ### 11.5 How to Use Extended Dataset

# %%
print('HOW TO USE THE EXTENDED DATASET')
print('='*70)

print('''
# 1. Add to your Kaggle notebook:
#    Click "Add Data" → Search "old-assyrian-extended-corpus"

# 2. Load the data:
corpus = pd.read_csv('/kaggle/input/old-assyrian-extended-corpus/akkadian_corpus.csv')
dictionary = pd.read_csv('/kaggle/input/old-assyrian-extended-corpus/akkadian_dictionary.csv')

# 3. Get parallel texts (same as train.csv but with more columns):
parallel = corpus[corpus['has_translation'] == True]

# 4. Get monolingual texts (for retrieval/pre-training):
monolingual = corpus[corpus['has_translation'] == False]

# 5. Filter by genre:
letters = corpus[corpus['genre_label'] == 'letter']

# 6. Get clean texts (no damaged sections):
clean = corpus[corpus['gap_count'] == 0]

# 7. Lookup logogram meanings:
logograms = dictionary[dictionary['entry_type'] == 'logogram']
meanings = logograms[logograms['known_meaning'].notna()]
''')

# %% [markdown]
# ### 11.6 Summary: Why Use Extended Dataset?

# %%
if EXTENDED_AVAILABLE:
    print('='*70)
    print('SUMMARY: WHY USE THE EXTENDED DATASET?')
    print('='*70)
    
    print('''
┌─────────────────────────────────────────────────────────────────────┐
│  BENEFIT                           │  HOW IT HELPS                  │
├─────────────────────────────────────────────────────────────────────┤''')
    print(f'│  +{len(monolingual):,} monolingual texts            │  Better TF-IDF retrieval       │')
    print(f'│  +{len(corpus.columns)-len(train.columns)} metadata columns                │  Filter by genre/quality       │')
    print(f'│  +{len(in_train):,} words with frequency          │  Identify important words      │')
    print(f'│  +{len(with_meaning)} logogram meanings               │  Understand Sumerograms        │')
    print('│  CDLI IDs included                 │  Cross-reference research      │')
    print('│  gap_count field                   │  Filter damaged texts          │')
    print('└─────────────────────────────────────────────────────────────────────┘')
    
    print('\n⭐ If this extended dataset helps you, please upvote! ⭐')


