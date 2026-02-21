# %% [markdown]
# # 🐪 Deep Past Initiative: Exploratory Data Analysis
# 
# This notebook explores the **Old Assyrian → English** machine translation dataset for the *Deep Past Initiative* Kaggle challenge.  
# The goal here is to **understand every CSV**, inspect text formats, and prepare insights for later modeling.
# 
# We will:
# - Inspect all provided files and their schema  
# - Look closely at transliterations, translations, and lexicon info  
# - Visualize key statistics to guide model design  
# 
# > This notebook focuses on **clean, visual EDA** before touching any modeling.

# %%
import os
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 5)

# %%
OA_Lexicon = pd.read_csv("/kaggle/input/deep-past-initiative-machine-translation/OA_Lexicon_eBL.csv")
bibliography = pd.read_csv("/kaggle/input/deep-past-initiative-machine-translation/bibliography.csv")
publications = pd.read_csv("/kaggle/input/deep-past-initiative-machine-translation/publications.csv")
pub_text = pd.read_csv("/kaggle/input/deep-past-initiative-machine-translation/published_texts.csv")
sample_sub = pd.read_csv("/kaggle/input/deep-past-initiative-machine-translation/sample_submission.csv")
test_df = pd.read_csv("/kaggle/input/deep-past-initiative-machine-translation/test.csv")
train_df = pd.read_csv("/kaggle/input/deep-past-initiative-machine-translation/train.csv")

# %% [markdown]
# ## 1. Dataset Overview
# 
# We start by inspecting the **shape and columns** of each main CSV to understand what information is available and how tables relate to each other.
# 
# This section answers:
# - How many rows does each file contain?
# - What are the key columns and dtypes?
# - Are there obvious missing values?

# %%
datasets = {
    "train_df": train_df,
    "test_df": test_df,
    "sample_sub": sample_sub,
    "pub_text": pub_text,
    "publications": publications,
    "bibliography": bibliography,
    "OA_Lexicon": OA_Lexicon,
}

summary_rows = []
for name, df in datasets.items():
    summary_rows.append({
        "name": name,
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "columns": ", ".join(df.columns[:8]) + ("..." if df.shape[1] > 8 else ""),
    })

overview_df = pd.DataFrame(summary_rows).sort_values("name")
overview_df

# %%
print("=== train_df.info() ===")
train_df.info()
print("\n=== test_df.info() ===")
test_df.info()
print("\n=== pub_text.info() ===")
pub_text.info()
print("\n=== OA_Lexicon.info() ===")
OA_Lexicon.info()

# %% [markdown]
# ## 2. Exploring train and test texts
# 
# Here we inspect **lengths and sample rows** from `train_df` and `test_df` to understand:
# - How long the transliterations and translations are
# - How document-level `train_df` differs from sentence-level `test_df`

# %%
# Add basic length features
train_df["src_len_char"] = train_df["transliteration"].str.len()
train_df["tgt_len_char"] = train_df["translation"].str.len()
train_df["src_len_tok"] = train_df["transliteration"].str.split().str.len()
train_df["tgt_len_tok"] = train_df["translation"].str.split().str.len()

length_summary = train_df[["src_len_char", "tgt_len_char", "src_len_tok", "tgt_len_tok"]].describe().T
length_summary

# %%
# few rows to see raw formatting
pd.set_option("display.max_colwidth", 200)

print("=== Sample train_df rows ===")
display(train_df.sample(3, random_state=42))

print("\n=== test_df ===")
display(test_df)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(train_df["src_len_tok"], bins=40, ax=axes[0], color="tab:blue")
axes[0].set_title("Transliteration token length (train)")
axes[0].set_xlabel("Tokens")

sns.histplot(train_df["tgt_len_tok"], bins=40, ax=axes[1], color="tab:green")
axes[1].set_title("Translation token length (train)")
axes[1].set_xlabel("Tokens")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Anatomy of the transliterations
# 
# Old Assyrian transliterations contain:
# - Hyphenated syllables, determinatives in `{ }`, and line/gap markers  
# - Diacritics and special letters (š, Ṣ, Ṭ, ḫ, etc.)
# 
# Here we look at:
# - Most frequent tokens
# - How often special characters and brackets appear

# %%
import re
from collections import Counter

def get_token_counts(series, top_n=30):
    tokens = []
    for text in series.dropna():
        tokens.extend(text.split())
    counter = Counter(tokens)
    return pd.DataFrame(counter.most_common(top_n), columns=["token", "freq"])

top_tokens_train = get_token_counts(train_df["transliteration"], top_n=40)
top_tokens_train.head(20)

# %%
# Special-character and bracket statistics in transliteration
special_chars = ["š", "Š", "ṣ", "Ṣ", "ṭ", "Ṭ", "ḫ", "Ḫ", "{", "}", "<", ">", "[", "]", ":", "!", "?", "/"]

stats = []
for ch in special_chars:
    count = train_df["transliteration"].str.count(re.escape(ch)).sum()
    docs = (train_df["transliteration"].str.contains(re.escape(ch))).sum()
    stats.append({"char": ch, "total_occurrences": int(count), "docs_with_char": int(docs)})

special_df = pd.DataFrame(stats).sort_values("docs_with_char", ascending=False)
special_df

# %%
# top 25 tokens in transliteration
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_tokens_train.head(25),
    x="freq",
    y="token",
    color="tab:blue"
)
plt.title("Most frequent transliteration tokens (train)")
plt.xlabel("Frequency")
plt.ylabel("Token")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.1 Observations on transliteration tokens
# 
# - The most frequent tokens include function words (e.g. `a-na`, `ša`, `i-na`) and numeric / measure tokens such as `KÙ.BABBAR`, `ma-na`, `GÍN`, and digits.  
# - Diacritic characters like `š`, `ḫ`, and `ṣ` occur in the vast majority of documents, confirming that **Unicode-aware tokenization** is necessary.  
# - Modern scribal symbols like `{}`, `<>`, `!`, `?`, `/`, and `:` barely appear in `train_df`, suggesting that many formatting suggestions are already applied in this cleaned split.

# %% [markdown]
# ## 4. Linking training texts to published metadata
# 
# Here we ask:
# - How many training documents have entries in `published_texts.csv`?
# - What extra fields (genre, description, AICC translation) can be leveraged later?

# %%
train_with_meta = train_df.merge(pub_text, on="oare_id", how="left", suffixes=("", "_pub"))

n_with_meta = train_with_meta["online transcript"].notna().sum()
n_total = len(train_df)

print(f"Training docs with metadata in pub_text: {n_with_meta} / {n_total} "
      f"({n_with_meta / n_total:.1%})")

# few examples with metadata present
cols_to_show = [
    "oare_id", "transliteration", "translation",
    "label", "genre_label", "description", "AICC_translation", "online transcript"
]
display(train_with_meta[train_with_meta["online transcript"].notna()][cols_to_show].head(3))

# %% [markdown]
# ### 4.1 Metadata coverage insights
# 
# - Every training document links to a `published_texts` record, so **oare_id is a perfect bridge** between aligned translations and richer metadata.  
# - Fields like `genre_label` (e.g. *debt note, note, unknown*) and detailed `description` can support **domain-aware analysis** or later conditioning in the model.  
# - The `AICC_translation` URLs and `online transcript` links provide external, noisy machine translations and tablet images that could be mined or used for qualitative inspection.

# %% [markdown]
# ## 5. Lexicon coverage of training vocabulary
# 
# The `OA_Lexicon_eBL.csv` file lists Old Assyrian word forms, their normalized shapes, and lexemes, plus type labels (e.g. `word`, `PN`, `GN`).  
# Here we measure how much of the **training vocabulary** is explained by the lexicon.

# %%
# Build training vocabulary (case-sensitive tokens)
train_tokens = []
for text in train_df["transliteration"]:
    train_tokens.extend(text.split())

train_vocab = pd.Series(train_tokens).value_counts()
print(f"Unique tokens in train transliteration: {len(train_vocab)}")

# lexicon lookup on form
lex_forms = set(OA_Lexicon["form"].dropna().astype(str))

# coverage by exact form match
in_lex = train_vocab.index.isin(lex_forms)
coverage_form = in_lex.mean()
print(f"Fraction of train vocab found in lexicon by exact `form`: {coverage_form:.2%}")

# token frequency coverage
freq_coverage_form = train_vocab[in_lex].sum() / train_vocab.sum()
print(f"Fraction of total token occurrences covered by `form`: {freq_coverage_form:.2%}")

# %%
# Joining vocab with lexicon to inspect types of covered tokens
vocab_df = train_vocab.rename("freq").reset_index().rename(columns={"index": "token"})

lex_small = OA_Lexicon[["form", "type", "norm", "lexeme"]].dropna(subset=["form"])
vocab_lex = vocab_df.merge(lex_small, left_on="token", right_on="form", how="left")

# How many tokens get at least one lexicon entry?
covered = vocab_lex["type"].notna().mean()
print(f"Vocab tokens with at least one lexicon row: {covered:.2%}")

type_stats = (
    vocab_lex[vocab_lex["type"].notna()]
    .groupby("type")["freq"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
type_stats.head(10)

# %%
# Examples of unmapped but frequent tokens
unmapped = vocab_lex[vocab_lex["type"].isna()].head(20)
unmapped

# %% [markdown]
# ## 5. Lexicon coverage insights
# 
# About **70% of unique train tokens** and over **80% of all token occurrences** have a direct `form` match in the lexicon, and most of that coverage is regular words plus a substantial block of personal names, while unmapped tokens are mainly numerals and gap markers that the lexicon does not attempt to encode.
# 
# - Out of **11,761** unique transliteration tokens in `train_df`, **69.76%** appear in `OA_Lexicon.form`, covering **82.37%** of all token occurrences, so the lexicon explains the majority of the running text.  
# - When joining vocabulary to `OA_Lexicon`, **73.02%** of token types get at least one lexical entry, with frequencies dominated by `word` (**123,268** occurrences), followed by `PN` (personal names, **22,116**) and `GN` (geographic names, **842**), confirming that names are a large, distinct slice of the corpus.  
# - The most frequent **unmapped tokens** include `x`, `…`, numeric tokens like `1, 2, 5, 10, 0.5, 0.33333`, and markers such as `[...]` or `xx`, which correspond to gaps, uncertainties, and numerical quantities that are intentionally outside the lexicon’s scope.

# %% [markdown]
# ## 6. Publications and bibliography overview
# 
# The `publications.csv` file contains OCR text from ~900 PDFs, and `bibliography.csv` adds metadata (title, author, year, journal).  
# Here we measure how many pages and PDFs contain Akkadian transliterations and how these sources are distributed over time and venues.

# %%
# Basic counts
n_pages = len(publications)
n_pages_akk = publications["has_akkadian"].sum()
n_pdfs = publications["pdf_name"].nunique()
n_pdfs_akk = publications.loc[publications["has_akkadian"], "pdf_name"].nunique()

print(f"Total pages in publications.csv: {n_pages}")
print(f"Pages with has_akkadian=True: {n_pages_akk} "
      f"({n_pages_akk / n_pages:.1%})")
print(f"Unique PDFs: {n_pdfs}")
print(f"PDFs with any Akkadian pages: {n_pdfs_akk} "
      f"({n_pdfs_akk / n_pdfs:.1%})")

# %%
# Merge with bibliography to get year and journal
pub_pages = publications.merge(bibliography, on="pdf_name", how="left")

year_counts = (
    pub_pages.dropna(subset=["year"])
    .groupby("year")["pdf_name"]
    .nunique()
    .sort_index()
)

journal_counts = (
    pub_pages.dropna(subset=["journal"])
    .groupby("journal")["pdf_name"]
    .nunique()
    .sort_values(ascending=False)
    .head(15)
)

year_counts.head(), journal_counts.head()

# %%
#PDFs per year (Akkadian only vs all)

pdf_year_all = (
    pub_pages.dropna(subset=["year"])
    .groupby("year")["pdf_name"]
    .nunique()
    .rename("all_pdfs")
)

pdf_year_akk = (
    pub_pages[pub_pages["has_akkadian"]]
    .dropna(subset=["year"])
    .groupby("year")["pdf_name"]
    .nunique()
    .rename("akk_pdfs")
)

pdf_year = pd.concat([pdf_year_all, pdf_year_akk], axis=1).fillna(0)

plt.figure(figsize=(12, 5))
pdf_year.plot(kind="bar", ax=plt.gca())
plt.title("Number of PDFs per year (all vs with Akkadian pages)")
plt.xlabel("Year")
plt.ylabel("Number of PDFs")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Publications corpus insights
# 
# The OCR corpus in `publications.csv` is **huge**: 216,602 pages across 952 PDFs, but only **31,286 pages (14.4%)** are flagged with `has_akkadian=True`, so Akkadian content is relatively sparse in each volume.  
# Still, almost **half of all PDFs (435 / 952, 45.7%)** contain at least one Akkadian page, meaning a wide spread of potentially useful sources for mining additional parallel data.
# 
# The bibliography spans more than a century: early entries appear already in the **1880s–1900s**, and coverage grows strongly after mid‑20th century, with a visible peak in late‑20th / early‑21st‑century years in the bar plot.  
# A small set of journals contributes a large share of PDFs, led by *Journal of the British Institute of Archaeology at Ankara* (44 PDFs), *Iraq* (32), *Journal of Cuneiform Studies* (16), *Zeitschrift für Assyriologie* (15), and *JESHO* (15), which are natural starting points if manually inspecting OCR quality and alignment.

# %% [markdown]
# ## 7. What do OCR pages look like?
# 
# To understand how hard it will be to mine extra training data, we inspect a few pages where `has_akkadian=True`.  
# This reveals how Akkadian transliterations and their translations are mixed with running prose, headings, and references.

# %%
# Random sample of Akkadian-containing pages
akk_pages = publications[publications["has_akkadian"]].sample(5, random_state=42)
akk_pages[["pdf_name", "page"]]

# %%
# snippets of the OCR text
pd.set_option("display.max_colwidth", 400)

for i, row in akk_pages.iterrows():
    print("=" * 80)
    print(f"PDF: {row['pdf_name']} | page: {row['page']}")
    print("-" * 80)
    snippet = row["page_text"]
    # Take a manageable slice
    print(snippet[:2000])
    print("\n")

# %% [markdown]
# ## 8. Design notes for mining extra training data
# 
# The **publications corpus is very heterogeneous**, so careful filtering is needed before using it as additional training data.
# 
# - Many pages interleave **Akkadian (or related) transliterations** with immediate translations and long stretches of modern-language commentary.  
# - Some pages are almost entirely **German/French/English prose** about history or law with only a few cited lines, while others are **pure bibliography/reference lists** without usable transliteration–translation pairs.
# 
# A practical extraction strategy will likely need:
# 
# - **Akkadian line detection**: identify candidate transliteration lines by features like heavy use of diacritics (`š, ḫ, ṣ`), hyphenated syllables, and limited lowercase prose.  
# - **Neighbouring translation pairing**: for each detected transliteration block, search in the nearby lines/paragraphs for sentences in a modern language (initially English, maybe later via MT from French/German) to form noisy parallel pairs.  
# - **Page-level filtering**: skip pages that look like bibliographies (many author-year patterns, very short lines, no diacritics or hyphenated tokens) or general narrative text with almost no Akkadian markers.  
# 
# These heuristics can guide a later **semi-automatic mining pipeline** that augments the small clean `train_df` with additional, but noisier, sentence-level examples.

# %% [markdown]
# ## 9. Summary & key takeaways to start building your model
# 
# <div style="background:#0f172a; color:#e5e7eb; padding:18px 20px; border-radius:10px; border:1px solid #1f2937; font-size:14px;">
# 
# <table style="width:100%; border-collapse:collapse; table-layout:fixed;">
#   <thead>
#     <tr>
#       <th style="border-bottom:1px solid #4b5563; padding:8px; text-align:left; width:22%;">Aspect</th>
#       <th style="border-bottom:1px solid #4b5563; padding:8px; text-align:left;">Findings</th>
#       <th style="border-bottom:1px solid #4b5563; padding:8px; text-align:left; width:26%;">Implications</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <td style="border-bottom:1px solid #374151; padding:8px;">Train vs. test structure</td>
#       <td style="border-bottom:1px solid #374151; padding:8px;">
#         Train has 1,561 <strong>document-level</strong> pairs with median ≈49 source vs. 68 target tokens and long tails up to 700+ target tokens.  
#         Test is <strong>sentence-level</strong> with line ranges per tablet.
#       </td>
#       <td style="border-bottom:1px solid #374151; padding:8px;">
#         Need strategies to <strong>segment documents into sentences</strong> for training and to handle long sequences (chunking or long-context models).
#       </td>
#     </tr>
#     <tr>
#       <td style="border-bottom:1px solid #374151; padding:8px;">Transliteration anatomy</td>
#       <td style="border-bottom:1px solid #374151; padding:8px;">
#         Frequent tokens mix function words (<code>a-na</code>, <code>ša</code>, <code>i-na</code>), logograms (<code>KÙ.BABBAR</code>, <code>DUMU</code>, <code>GÍN</code>), and numeric markers.  
#         Diacritics like <code>š</code>, <code>ḫ</code>, <code>ṣ</code> appear in most documents; modern scribal punctuation is already rare.
#       </td>
#       <td style="border-bottom:1px solid #374151; padding:8px;">
#         Requires <strong>Unicode-aware tokenization</strong> and probably custom normalization rules; standard Latin-only tokenizers will underperform.
#       </td>
#     </tr>
#     <tr>
#       <td style="border-bottom:1px solid #374151; padding:8px;">Metadata linkage</td>
#       <td style="border-bottom:1px solid #374151; padding:8px;">
#         Every train row has a matching entry in <code>published_texts.csv</code> via <code>oare_id</code>, with fields like <code>genre_label</code>, descriptions, and external links.
#       </td>
#       <td style="border-bottom:1px solid #374151; padding:8px;">
#         Enables <strong>genre-aware analysis</strong>, curriculum ideas, and easy sampling of specific document types (e.g. debt notes vs. letters).
#       </td>
#     </tr>
#     <tr>
#       <td style="border-bottom:1px solid #374151; padding:8px;">Lexicon coverage</td>
#       <td style="border-bottom:1px solid #374151; padding:8px;">
#         Lexicon covers ≈<strong>70% of unique tokens</strong> and ≈<strong>82% of token occurrences</strong>; mapped tokens are dominated by <code>word</code>, with substantial <code>PN</code> and <code>GN</code>.  
#         Unmapped tokens are mainly numerals, gaps (<code>x</code>, <code>…</code>, <code>[...]</code>), and placeholders.
#       </td>
#       <td style="border-bottom:1px solid #374151; padding:8px;">
#         Lexicon is a strong prior for <strong>morphology, lemmatization, and proper-name handling</strong>; unmapped tokens can be normalized or treated as numeric / gap classes.
#       </td>
#     </tr>
#     <tr>
#       <td style="border-bottom:1px solid #374151; padding:8px;">Publications corpus scale</td>
#       <td style="border-bottom:1px solid #374151; padding:8px;">
#         216k pages across 952 PDFs; only 14.4% of pages and ~46% of PDFs contain any Akkadian.  
#         A handful of journals contribute a large fraction of relevant material.
#       </td>
#       <td style="border-bottom:1px solid #374151; padding:8px;">
#         Mining extra data is feasible but requires <strong>aggressive filtering</strong> to avoid drowning in non-parallel text.
#       </td>
#     </tr>
#     <tr>
#       <td style="padding:8px;">OCR page structure</td>
#       <td style="padding:8px;">
#         Pages mix transliteration lines, inline translations, multi-language commentary, and pure bibliography.  
#         Layout conventions differ widely between works and decades.
#       </td>
#       <td style="padding:8px;">
#         Any automatic mining pipeline should combine <strong>Akkadian-line detection</strong>, nearby-sentence pairing, and bibliographic-page filtering before using mined pairs for training.
#       </td>
#     </tr>
#   </tbody>
# </table>
# 
# </div>
# 

# %% [markdown]
# ## 🙏 Thanks for checking this notebook!
# 
# <div style="margin-top:10px; padding:18px 20px; border-radius:10px; border:1px solid #e5e7eb; background:linear-gradient(135deg,#0f172a,#111827); color:#f9fafb; text-align:center; font-size:15px;">
# 
#   <div style="font-size:20px; font-weight:600; margin-bottom:8px;">
#     Thanks for exploring this EDA!
#   </div>
# 
#   <div style="max-width:700px; margin:0 auto 10px auto; line-height:1.6;">
#     If this notebook helped you understand the Deep Past dataset or prepare your
#     machine translation model, consider giving it an <strong>upvote</strong> on Kaggle.
#     Your support helps others discover it and motivates further updates 🚀
#   </div>
# 
#   <div style="font-size:13px; color:#9ca3af;">
#     Feel free to fork, extend the analyses, or adapt the code for your own experiments.
#   </div>
# 
# </div>


