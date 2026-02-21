# %% [markdown]
# ## Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

DATA_PATH = '/kaggle/input/deep-past-initiative-machine-translation'

# %% [markdown]
# ## 1. Data Overview
# 
# Let's start by loading all datasets and understanding their structure.

# %%
# Load all datasets
train = pd.read_csv(f'{DATA_PATH}/train.csv')
test = pd.read_csv(f'{DATA_PATH}/test.csv')
sample_sub = pd.read_csv(f'{DATA_PATH}/sample_submission.csv')
published_texts = pd.read_csv(f'{DATA_PATH}/published_texts.csv')
lexicon = pd.read_csv(f'{DATA_PATH}/OA_Lexicon_eBL.csv')
bibliography = pd.read_csv(f'{DATA_PATH}/bibliography.csv')
publications = pd.read_csv(f'{DATA_PATH}/publications.csv')

print("="*70)
print("📊 DATASET OVERVIEW")
print("="*70)

datasets = {
    'train.csv': train,
    'test.csv': test,
    'sample_submission.csv': sample_sub,
    'published_texts.csv': published_texts,
    'OA_Lexicon_eBL.csv': lexicon,
    'bibliography.csv': bibliography,
    'publications.csv': publications
}

summary_data = []
for name, df in datasets.items():
    summary_data.append({
        'Dataset': name,
        'Rows': f"{len(df):,}",
        'Columns': len(df.columns),
        'Memory (MB)': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}"
    })

summary_df = pd.DataFrame(summary_data)
display(summary_df)

# %%
# Visual overview of dataset sizes
fig, ax = plt.subplots(figsize=(10, 5))

sizes = [len(train), len(test), len(published_texts), len(lexicon), len(bibliography), len(publications)]
names = ['Training\nPairs', 'Test\nSentences', 'Published\nTexts', 'Lexicon\nEntries', 'Bibliography', 'Publication\nPages']
colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c']

bars = ax.bar(names, sizes, color=colors, edgecolor='white', linewidth=2)

# Add value labels
for bar, size in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.01, 
            f'{size:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Number of Records', fontsize=12)
ax.set_title('📊 Dataset Sizes at a Glance', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(sizes) * 1.15)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Training Data Deep Dive
# 
# The training data contains **document-level** parallel pairs of transliterated Akkadian and English translations.

# %%
print("🔍 TRAINING DATA STRUCTURE")
print("="*70)
print(f"\nShape: {train.shape}")
print(f"\nColumns: {train.columns.tolist()}")
print(f"\nData types:\n{train.dtypes}")
print(f"\nNull values: {train.isnull().sum().sum()}")

# %%
# Show example pairs
print("\n📜 EXAMPLE TRANSLATION PAIRS")
print("="*70)

for i in [0, 100, 500]:
    print(f"\n{'─'*70}")
    print(f"📄 Document {i} (oare_id: {train.iloc[i]['oare_id'][:8]}...)")
    print(f"{'─'*70}")
    print(f"\n🔤 TRANSLITERATION:")
    print(f"{train.iloc[i]['transliteration'][:300]}...")
    print(f"\n🇬🇧 TRANSLATION:")
    print(f"{train.iloc[i]['translation'][:300]}...")

# %%
# Calculate text statistics
train['translit_chars'] = train['transliteration'].str.len()
train['trans_chars'] = train['translation'].str.len()
train['translit_words'] = train['transliteration'].str.split().str.len()
train['trans_words'] = train['translation'].str.split().str.len()
train['char_ratio'] = train['trans_chars'] / train['translit_chars']
train['word_ratio'] = train['trans_words'] / train['translit_words']

# Statistics table
stats_df = pd.DataFrame({
    'Metric': ['Characters', 'Words'],
    'Transliteration (mean)': [f"{train['translit_chars'].mean():.0f}", f"{train['translit_words'].mean():.0f}"],
    'Transliteration (median)': [f"{train['translit_chars'].median():.0f}", f"{train['translit_words'].median():.0f}"],
    'Translation (mean)': [f"{train['trans_chars'].mean():.0f}", f"{train['trans_words'].mean():.0f}"],
    'Translation (median)': [f"{train['trans_chars'].median():.0f}", f"{train['trans_words'].median():.0f}"],
    'Ratio (trans/translit)': [f"{train['char_ratio'].mean():.2f}x", f"{train['word_ratio'].mean():.2f}x"]
})

print("\n📊 TEXT LENGTH STATISTICS")
display(stats_df)

# %%
# Visualize length distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Character lengths
axes[0, 0].hist(train['translit_chars'], bins=50, alpha=0.7, color='#3498db', edgecolor='white')
axes[0, 0].axvline(train['translit_chars'].median(), color='red', linestyle='--', label=f'Median: {train["translit_chars"].median():.0f}')
axes[0, 0].set_xlabel('Character Length')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('📝 Transliteration Length (Characters)', fontweight='bold')
axes[0, 0].legend()

axes[0, 1].hist(train['trans_chars'], bins=50, alpha=0.7, color='#2ecc71', edgecolor='white')
axes[0, 1].axvline(train['trans_chars'].median(), color='red', linestyle='--', label=f'Median: {train["trans_chars"].median():.0f}')
axes[0, 1].set_xlabel('Character Length')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('📝 Translation Length (Characters)', fontweight='bold')
axes[0, 1].legend()

# Word lengths
axes[1, 0].hist(train['translit_words'], bins=50, alpha=0.7, color='#9b59b6', edgecolor='white')
axes[1, 0].axvline(train['translit_words'].median(), color='red', linestyle='--', label=f'Median: {train["translit_words"].median():.0f}')
axes[1, 0].set_xlabel('Word Count')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('📊 Transliteration Length (Words)', fontweight='bold')
axes[1, 0].legend()

axes[1, 1].hist(train['trans_words'], bins=50, alpha=0.7, color='#e74c3c', edgecolor='white')
axes[1, 1].axvline(train['trans_words'].median(), color='red', linestyle='--', label=f'Median: {train["trans_words"].median():.0f}')
axes[1, 1].set_xlabel('Word Count')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('📊 Translation Length (Words)', fontweight='bold')
axes[1, 1].legend()

plt.tight_layout()
plt.suptitle('Training Data Length Distributions', fontsize=14, fontweight='bold', y=1.02)
plt.show()

# %%
# Source vs Target length relationship
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot
axes[0].scatter(train['translit_words'], train['trans_words'], alpha=0.4, c='#3498db', s=20)
axes[0].plot([0, 200], [0, 200*train['word_ratio'].mean()], 'r--', 
             label=f'Avg ratio: {train["word_ratio"].mean():.2f}x')
axes[0].set_xlabel('Transliteration Words')
axes[0].set_ylabel('Translation Words')
axes[0].set_title('🔗 Source vs Target Word Count', fontweight='bold')
axes[0].legend()

# Ratio distribution
axes[1].hist(train['word_ratio'], bins=50, alpha=0.7, color='#9b59b6', edgecolor='white')
axes[1].axvline(train['word_ratio'].mean(), color='red', linestyle='--', 
                label=f'Mean: {train["word_ratio"].mean():.2f}x')
axes[1].set_xlabel('Word Ratio (Translation / Transliteration)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('📈 Translation Expansion Ratio', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"\n💡 Key Insight: Translations are on average {train['word_ratio'].mean():.2f}x longer than transliterations")
print(f"   This reflects Akkadian's morphological complexity - one word encodes multiple English words")

# %% [markdown]
# ## 3. 🔤 Transliteration Format Analysis
# 
# Understanding the special characters and patterns in transliterations is **critical** for preprocessing.

# %%
# Analyze transliteration patterns
all_translit = ' '.join(train['transliteration'].tolist())

print("🔍 TRANSLITERATION FORMAT ANALYSIS")
print("="*70)

# 1. Hyphenation
words = all_translit.split()
hyphenated = [w for w in words if '-' in w]
print(f"\n📌 HYPHENATION:")
print(f"   Total tokens: {len(words):,}")
print(f"   Hyphenated tokens: {len(hyphenated):,} ({100*len(hyphenated)/len(words):.1f}%)")
print(f"   Sample: {hyphenated[:5]}")

# 2. ALL CAPS (Sumerian logograms)
caps_pattern = re.compile(r'\b[A-ZŠṢṬḪ][A-ZŠṢṬḪ₀-₉\.]+\b')
caps_words = caps_pattern.findall(all_translit)
caps_counts = Counter(caps_words)
print(f"\n📌 SUMERIAN LOGOGRAMS (ALL CAPS):")
print(f"   Total occurrences: {len(caps_words):,}")
print(f"   Unique logograms: {len(caps_counts):,}")
print(f"   Top 10: {caps_counts.most_common(10)}")

# 3. Determinatives - check if present
determinatives = re.findall(r'\{[^}]+\}', all_translit)
parens_det = re.findall(r'\([^)]+\)', all_translit)
print(f"\n📌 DETERMINATIVES:")
print(f"   Curly brackets {{...}}: {len(determinatives)}")
print(f"   Parentheses (...): {len(parens_det)}")
if parens_det:
    print(f"   Sample parentheses: {list(set(parens_det))[:10]}")

# 4. Subscript numbers
subscripts = re.findall(r'[a-zšṣṭḫ]+[₀-₉]+', all_translit)
subscript_counts = Counter(subscripts)
print(f"\n📌 SUBSCRIPT NUMBERS:")
print(f"   Total occurrences: {len(subscripts):,}")
print(f"   Unique forms: {len(subscript_counts):,}")
print(f"   Top 10: {subscript_counts.most_common(10)}")

# 5. Fractions/decimals (representing ancient measures)
fractions = re.findall(r'\d+\.\d+', all_translit)
print(f"\n📌 FRACTIONS/DECIMALS:")
print(f"   Total occurrences: {len(fractions):,}")
print(f"   Sample values: {list(set(fractions))[:10]}")

# 6. Special punctuation
print(f"\n📌 SPECIAL PUNCTUATION:")
print(f"   Square brackets []: {all_translit.count('[') + all_translit.count(']')}")
print(f"   Parentheses (): {all_translit.count('(') + all_translit.count(')')}")
print(f"   Half brackets ˹˺: {all_translit.count('˹') + all_translit.count('˺')}")
print(f"   Question marks: {all_translit.count('?')}")
print(f"   Exclamation marks: {all_translit.count('!')}")

# %%
# Visualize Sumerian logogram distribution
top_logograms = caps_counts.most_common(20)

fig, ax = plt.subplots(figsize=(12, 6))
names = [x[0] for x in top_logograms]
counts = [x[1] for x in top_logograms]

bars = ax.barh(names[::-1], counts[::-1], color='#e74c3c', edgecolor='white')
ax.set_xlabel('Frequency', fontsize=12)
ax.set_title('📊 Top 20 Sumerian Logograms in Training Data', fontsize=14, fontweight='bold')

# Add value labels
for bar, count in zip(bars, counts[::-1]):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2, 
            f'{count:,}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.show()

print("\n💡 Key Logograms Explained:")
print("   • KÙ.BABBAR = Silver (most common commodity)")
print("   • DUMU = 'son (of)' - patronymic indicator")
print("   • KIŠIB = Seal (witness markers)")
print("   • IŠTAR = The goddess Ishtar")
print("   • AN.NA = Tin (key trade good)")

# %% [markdown]
# ## 4. Translation Analysis
# 
# Let's analyze the English translations to understand vocabulary and patterns.

# %%
# Analyze translations
all_trans = ' '.join(train['translation'].tolist())

# Word frequency
trans_words = re.findall(r'\b[a-z]+\b', all_trans.lower())
word_counts = Counter(trans_words)

print("🔍 TRANSLATION VOCABULARY ANALYSIS")
print("="*70)
print(f"\n📊 Basic Statistics:")
print(f"   Total word tokens: {len(trans_words):,}")
print(f"   Unique words: {len(word_counts):,}")
print(f"   Vocabulary density: {len(word_counts)/len(trans_words)*100:.2f}%")

print(f"\n📝 Top 30 Most Common Words:")
for i, (word, count) in enumerate(word_counts.most_common(30), 1):
    print(f"   {i:2d}. {word:15s} : {count:,}")

# %%
# Proper nouns analysis
proper_pattern = re.compile(r'\b[A-ZŠṢṬḪ][a-zšṣṭḫāēīūâêîû]+(?:-[A-Za-zšṣṭḫāēīūâêîû]+)*\b')
proper_nouns = proper_pattern.findall(all_trans)
proper_counts = Counter(proper_nouns)

# Filter out sentence starters
sentence_starters = {'The', 'If', 'To', 'From', 'He', 'My', 'In', 'As', 'They', 'We', 'You', 'This', 'That', 'When', 'After', 'Before'}
proper_counts_filtered = {k: v for k, v in proper_counts.items() if k not in sentence_starters}

print("\n📌 PROPER NOUNS (Names & Places):")
print(f"   Total occurrences: {len(proper_nouns):,}")
print(f"   Unique proper nouns: {len(proper_counts_filtered):,}")

# Top proper nouns
top_proper = sorted(proper_counts_filtered.items(), key=lambda x: x[1], reverse=True)[:20]

fig, ax = plt.subplots(figsize=(12, 6))
names = [x[0] for x in top_proper]
counts = [x[1] for x in top_proper]

bars = ax.barh(names[::-1], counts[::-1], color='#9b59b6', edgecolor='white')
ax.set_xlabel('Frequency', fontsize=12)
ax.set_title('📊 Top 20 Proper Nouns in Translations', fontsize=14, fontweight='bold')

for bar, count in zip(bars, counts[::-1]):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
            f'{count:,}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.show()

print("\n💡 These are primarily merchant names - this was a trading community!")

# %%
# Common translation phrases
print("\n📌 COMMON TRANSLATION PATTERNS:")
print("="*70)

common_phrases = [
    ('son of', 'Patronymic naming'),
    ('shekels of silver', 'Currency/payment'),
    ('minas of silver', 'Larger currency amounts'),
    ('seal of', 'Witness/authentication'),
    ('witnessed by', 'Legal witnesses'),
    ('will pay', 'Debt obligations'),
    ('owes', 'Debt statements'),
    ('from the week of', 'Time reference'),
    ('in the eponymy of', 'Year dating'),
    ('tin', 'Key trade commodity'),
    ('textiles', 'Key trade commodity'),
    ('caravan', 'Trade logistics'),
]

print(f"\n{'Phrase':<25} {'Count':>8}  Purpose")
print("-"*60)
for phrase, purpose in common_phrases:
    count = all_trans.lower().count(phrase.lower())
    print(f"{phrase:<25} {count:>8,}  {purpose}")

print("\n💡 These patterns reveal the commercial nature of the texts:")
print("   • Debt notes and IOUs")
print("   • Witnessed contracts")
print("   • Trade in silver, tin, and textiles")
print("   • Caravan logistics")

# %% [markdown]
# ## 5. 🧪 Test Data Analysis
# 
# ⚠️ **Critical**: The test data has a **different format** than training data!

# %%
print("🧪 TEST DATA STRUCTURE")
print("="*70)
print(f"\nShape: {test.shape}")
print(f"\nColumns: {test.columns.tolist()}")
print(f"\n⚠️  NOTE: This is PLACEHOLDER data. Real test has ~4,000 sentences from ~400 documents.")

display(test)

# %%
# Highlight the train/test mismatch
print("\n⚠️  CRITICAL: TRAIN VS TEST FORMAT MISMATCH")
print("="*70)

comparison = pd.DataFrame({
    'Aspect': ['Alignment Level', 'Avg Length (words)', 'Structure', 'Line Numbers'],
    'Training Data': ['Document-level', '~58 words', 'Complete tablets', 'Not provided'],
    'Test Data': ['Sentence-level', '~21 words (est.)', 'Individual sentences', 'line_start, line_end provided']
})

display(comparison)

print("\n💡 IMPLICATIONS:")
print("   1. You're training on DOCUMENTS but predicting SENTENCES")
print("   2. Need strategy for sentence segmentation OR")
print("   3. Create synthetic sentence-level training pairs")
print("   4. Line numbers in test can help with alignment")

# %%
# Sample submission format
print("\n📝 SUBMISSION FORMAT:")
print("="*70)
display(sample_sub)

print("\n💡 Each test sentence needs ONE English translation.")

# %% [markdown]
# ## 6. 📚 Supplementary Data: Published Texts
# 
# This dataset contains **8,000 transliterations** with metadata - but **no translations**. 
# Useful for language modeling and understanding the domain.

# %%
print("📚 PUBLISHED TEXTS ANALYSIS")
print("="*70)
print(f"\nShape: {published_texts.shape}")
print(f"\nColumns: {published_texts.columns.tolist()}")

# Check overlap with training
train_ids = set(train['oare_id'])
pub_ids = set(published_texts['oare_id'])
overlap = train_ids & pub_ids

print(f"\n📊 OVERLAP ANALYSIS:")
print(f"   Training documents: {len(train_ids):,}")
print(f"   Published texts: {len(pub_ids):,}")
print(f"   Overlap (in both): {len(overlap):,}")
print(f"   Additional texts without translations: {len(pub_ids - train_ids):,}")

# %%
# Genre distribution
genre_counts = published_texts['genre_label'].value_counts()

fig, ax = plt.subplots(figsize=(12, 6))

# Top 15 genres
top_genres = genre_counts.head(15)
colors = plt.cm.Spectral(np.linspace(0, 1, len(top_genres)))

bars = ax.barh(top_genres.index[::-1], top_genres.values[::-1], color=colors[::-1], edgecolor='white')
ax.set_xlabel('Number of Texts', fontsize=12)
ax.set_title('📊 Genre Distribution of Old Assyrian Texts', fontsize=14, fontweight='bold')

for bar, count in zip(bars, top_genres.values[::-1]):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2, 
            f'{count:,}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.show()

print("\n💡 Key Genres:")
print("   • Letters - Personal and business correspondence")
print("   • Debt notes - IOUs and financial obligations")
print("   • Agreements/Contracts - Legal documents")
print("   • Unknown - Many texts haven't been classified")

# %%
# Compare formatting between train and published_texts
print("\n🔍 FORMATTING DIFFERENCES: Train vs Published Texts")
print("="*70)

# Find a common text
common_id = list(overlap)[0]
train_text = train[train['oare_id'] == common_id]['transliteration'].iloc[0][:200]
pub_text = published_texts[published_texts['oare_id'] == common_id]['transliteration'].iloc[0][:200]

print(f"\nSame document in both datasets:")
print(f"\n📄 TRAIN.CSV:")
print(f"   {train_text}...")
print(f"\n📄 PUBLISHED_TEXTS.CSV:")
print(f"   {pub_text}...")

# Check for determinatives and gaps
pub_all = ' '.join(published_texts['transliteration'].dropna().tolist())
pub_determinatives = re.findall(r'\{[^}]+\}', pub_all)
pub_gaps = re.findall(r'<[^>]+>', pub_all)

print(f"\n📌 Published texts special markers:")
print(f"   Determinatives {{d}}, {{ki}}, etc.: {len(pub_determinatives):,}")
print(f"   Gap markers <gap>, <big_gap>: {len(pub_gaps):,}")
print(f"\n⚠️  Training data has DIFFERENT formatting:")
print(f"   - Determinatives as (d) instead of {{d}}")
print(f"   - No gap markers")

# %% [markdown]
# ## 7. 📖 Lexicon Analysis
# 
# The lexicon contains **39,000+ entries** mapping transliterated forms to normalized/lemmatized versions.

# %%
print("📖 LEXICON ANALYSIS")
print("="*70)
print(f"\nShape: {lexicon.shape}")
print(f"\nColumns: {lexicon.columns.tolist()}")

# Type distribution
type_counts = lexicon['type'].value_counts()
print(f"\n📊 ENTRY TYPES:")
for entry_type, count in type_counts.items():
    print(f"   {entry_type}: {count:,} ({100*count/len(lexicon):.1f}%)")

# %%
# Visualize lexicon composition
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
colors = ['#3498db', '#e74c3c', '#2ecc71']
explode = (0.02, 0.02, 0.02)
axes[0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
            colors=colors, explode=explode, shadow=True, startangle=90)
axes[0].set_title('📊 Lexicon Entry Types', fontsize=14, fontweight='bold')

# Sample entries
axes[1].axis('off')
sample_text = "📚 Sample Lexicon Entries\n\n"
sample_text += "WORDS:\n"
for _, row in lexicon[lexicon['type'] == 'word'].head(3).iterrows():
    sample_text += f"  {row['form']} → {row['lexeme']}\n"
sample_text += "\nPERSONAL NAMES (PN):\n"
for _, row in lexicon[lexicon['type'] == 'PN'].head(3).iterrows():
    sample_text += f"  {row['form']} → {row['lexeme']}\n"
sample_text += "\nGEOGRAPHIC NAMES (GN):\n"
for _, row in lexicon[lexicon['type'] == 'GN'].head(3).iterrows():
    sample_text += f"  {row['form']} → {row['lexeme']}\n"

axes[1].text(0.1, 0.9, sample_text, transform=axes[1].transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# %%
# Lexicon coverage of training data
print("\n🔍 LEXICON COVERAGE OF TRAINING DATA")
print("="*70)

# Get unique words from training
train_words = set(all_translit.split())
train_words_lower = set(w.lower() for w in train_words)

# Get lexicon forms
lex_forms = set(lexicon['form'].dropna().tolist())
lex_forms_lower = set(f.lower() for f in lex_forms)

# Calculate coverage
exact_matches = train_words & lex_forms
case_matches = train_words_lower & lex_forms_lower

print(f"\n📊 Coverage Statistics:")
print(f"   Unique tokens in training: {len(train_words):,}")
print(f"   Lexicon forms: {len(lex_forms):,}")
print(f"   Exact matches: {len(exact_matches):,} ({100*len(exact_matches)/len(train_words):.1f}%)")
print(f"   Case-insensitive matches: {len(case_matches):,} ({100*len(case_matches)/len(train_words_lower):.1f}%)")

print(f"\n💡 The lexicon covers ~70% of training vocabulary")
print(f"   This is especially useful for proper noun normalization!")

# %% [markdown]
# ## 8. 📰 Publications Mining Potential
# 
# The `publications.csv` contains OCR'd text from **878 scholarly publications** with embedded translations!

# %%
print("📰 PUBLICATIONS ANALYSIS")
print("="*70)
print(f"\nShape: {publications.shape}")
print(f"\nColumns: {publications.columns.tolist()}")

# Basic stats
print(f"\n📊 Basic Statistics:")
print(f"   Total pages: {len(publications):,}")
print(f"   Unique PDFs: {publications['pdf_name'].nunique():,}")
print(f"   Pages with Akkadian: {publications['has_akkadian'].sum():,} ({100*publications['has_akkadian'].mean():.1f}%)")

# %%
# Sample page content
print("\n📄 SAMPLE PAGE CONTENT:")
print("="*70)

# Find a page with Akkadian
akkadian_page = publications[publications['has_akkadian'] == True].iloc[0]
print(f"\nPDF: {akkadian_page['pdf_name'][:60]}...")
print(f"Page: {akkadian_page['page']}")
print(f"\nContent (first 1000 chars):")
print("-"*70)
print(akkadian_page['page_text'][:1000])

# %%
# Analyze bibliography for languages
print("\n📚 BIBLIOGRAPHY ANALYSIS")
print("="*70)
print(f"\nTotal publications: {len(bibliography):,}")

# Year distribution
bibliography['year_clean'] = pd.to_numeric(bibliography['year'].str[:4], errors='coerce')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Year distribution
year_counts = bibliography['year_clean'].dropna().astype(int)
axes[0].hist(year_counts, bins=30, color='#3498db', edgecolor='white', alpha=0.7)
axes[0].set_xlabel('Publication Year')
axes[0].set_ylabel('Number of Publications')
axes[0].set_title('📅 Publication Year Distribution', fontweight='bold')

# Journal distribution
journal_counts = bibliography['journal'].value_counts().head(10)
axes[1].barh(journal_counts.index[::-1], journal_counts.values[::-1], color='#9b59b6', edgecolor='white')
axes[1].set_xlabel('Number of Publications')
axes[1].set_title('📰 Top 10 Journals', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n💡 The publications span over 100 years of scholarship!")
print("   Translations may be in English, German, French, or Turkish.")

# %%
# Estimate potential additional training data
print("\n🎯 POTENTIAL DATA MINING OPPORTUNITY")
print("="*70)

pages_with_akkadian = publications['has_akkadian'].sum()
avg_page_len = publications['page_text'].str.len().mean()

print(f"\n📊 Estimation:")
print(f"   Pages with Akkadian text: {pages_with_akkadian:,}")
print(f"   Average page length: {avg_page_len:,.0f} characters")
print(f"   Total Akkadian content: ~{pages_with_akkadian * avg_page_len / 1e6:.1f}M characters")

print(f"\n💡 IF you can extract translations from these publications:")
print(f"   - Could potentially 5-10x your training data")
print(f"   - Translations in multiple languages (need translation)")
print(f"   - Alignment with published_texts.csv using document IDs")
print(f"\n⚠️  This is a MAJOR data engineering challenge but potentially game-changing!")

# %%
print("📌 KEY DATA CHALLENGES")
print("="*70)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ULTRA LOW RESOURCE: Only 1,561 training pairs
2. TRAIN/TEST MISMATCH: Training=documents, Test=sentences
3. MORPHOLOGICAL COMPLEXITY: 1.56x word expansion ratio
4. PROPER NOUNS: 13,000+ personal names to handle
""")

# %%



