# %% [markdown]
# # Deep Past Initiative – Machine Translation (Inference Notebook)
# 
# I noticed that the English text after translation sometimes sounded unnatural or awkward.
# To address this issue, I have shared code that applies LLM-based post-processing to improve the quality of the translated text.
# 
# The current implementation is based on the publicly available Best Score code, with additional post-processing using Gemma-3.
# 
# ###  Update history:
# 
# - [Base Notebook](https://www.kaggle.com/code/takamichitoda/dpc-starter-infer)
# 	- Public Score: 26.6
# - [Version 5](https://www.kaggle.com/code/takamichitoda/dpc-infer-with-post-processing-by-llm?scriptVersionId=287804751): 
# 	- Public Score: 28.9
# 	- Use the Best Score model at the time. 
# 	- But it have been deleted.
# - [Version 7](https://www.kaggle.com/code/takamichitoda/dpc-infer-with-post-processing-by-llm?scriptVersionId=290075269): 
# 	- Public Score: 30.3
# 	- Use the Best Score model at the time.
# 	- model -> [byt5-base-32.6-third](https://www.kaggle.com/datasets/jeanjean111/byt5-base-big-data2)
# - [Version 10](https://www.kaggle.com/code/takamichitoda/dpc-infer-with-post-processing-by-llm?scriptVersionId=290148346)
# 	- Public Score: 32.6
# 	- Fix Prompt.
# - [Version 12](https://www.kaggle.com/code/takamichitoda/dpc-infer-with-post-processing-by-llm?scriptVersionId=292147659)
# 	- Public Score: 
# 	- Use the Best Score model at the time.
# 	- weight AVG -> [byt5-base-big-data2](https://www.kaggle.com/datasets/jeanjean111/byt5-base-big-data2), [train-gap-all-2](https://www.kaggle.com/datasets/qifeihhh666/train-gap-all-2), [byt5-akkadian-model](https://www.kaggle.com/datasets/llkh0a/byt5-akkadian-model)
#       - Reference code is [here](https://www.kaggle.com/code/yongsukprasertsuk/deep-past-challenge-weight-averaging).
#     - Shift the post-processing from an LLM-centric approach to a more conservative, consistency-focused approach centered on the dictionary (OA_Lexicon) and train references (translation memory).
# 
# 
# 
# If you fork this code, don’t forget to upvote the `qifeihhh666`, `jeanjean111`, and `llkh0a` datasets shared by their awesome authors. Let’s support the spirit of contribution!  
# 👉 [byt5-base-big-data2](https://www.kaggle.com/datasets/jeanjean111/byt5-base-big-data2)  
# 👉 [train-gap-all-2](https://www.kaggle.com/datasets/qifeihhh666/train-gap-all-2)  
# 👉 [byt5-akkadian-model](https://www.kaggle.com/datasets/llkh0a/byt5-akkadian-model)   

# %%
import re
import gc
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

# %%
MODEL1_PATH = "/kaggle/input/byt5-base-big-data2"
MODEL2_PATH = "/kaggle/input/byt5-akkadian-model"
MODEL3_PATH = "/kaggle/input/train-gap-all-2/byt5-base-akkadian_gap_setence2"

TEST_DATA_PATH = "/kaggle/input/deep-past-initiative-machine-translation/test.csv"
BATCH_SIZE = 4
MAX_LENGTH = 512
MAX_NEW_TOKENS = 512
BATCH_SIZE = 8
NUM_BEAMS = 10
LENGTH_PENALTY = 1.08
EARLY_STOPPING = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# =========================
# Load models
# =========================
print("Loading models...")

m1 = AutoModelForSeq2SeqLM.from_pretrained(MODEL1_PATH)
m2 = AutoModelForSeq2SeqLM.from_pretrained(MODEL2_PATH)
m3 = AutoModelForSeq2SeqLM.from_pretrained(MODEL3_PATH)

sd1, sd2, sd3 = m1.state_dict(), m2.state_dict(), m3.state_dict()

# =========================
# Weighted checkpoint averaging
# =========================
perf1, perf2, perf3 = 0.98, 1.00, 0.40
total = perf1 + perf2 + perf3
w1, w2, w3 = perf1/total, perf2/total, perf3/total

print(f"Weights → w1={w1:.3f}, w2={w2:.3f}, w3={w3:.3f}")

final_sd = sd2.copy()
for k in final_sd:
    if k in sd1 and k in sd3:
        final_sd[k] = w1 * sd1[k] + w2 * sd2[k] + w3 * sd3[k]
    elif k in sd1:
        final_sd[k] = w1 * sd1[k] + (w2 + w3) * sd2[k]
    elif k in sd3:
        final_sd[k] = w3 * sd3[k] + (w1 + w2) * sd2[k]

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL2_PATH)
model.load_state_dict(final_sd)
model.to(DEVICE).eval().float()

tokenizer = AutoTokenizer.from_pretrained(MODEL2_PATH)

del m1, m2, m3, sd1, sd2, sd3
gc.collect()
torch.cuda.empty_cache()

# %%
# =========================
# Gap normalization (VERY IMPORTANT)
# =========================
def replace_gaps(text):
    if pd.isna(text):
        return text
    text = str(text)
    text = re.sub(r'\.3(?:\s+\.3)+\.{3}(?:\s+\.{3})+', '<big_gap>', text)
    text = re.sub(r'\.3(?:\s+\.3)+\.{3}(?:\s+\.{3})+', '<big_gap>', text)
    text = re.sub(r'\.{3}(?:\s+\.{3})+', '<big_gap>', text)
    text = re.sub(r'xx', '<gap>', text)
    text = re.sub(r' x ', ' <gap> ', text)
    text = re.sub(r'……', '<big_gap>', text)
    text = re.sub(r'\.\.\.\.\.\.', '<big_gap>', text)
    text = re.sub(r'…', '<big_gap>', text)
    text = re.sub(r'\.\.\.', '<big_gap>', text)
    return text

# %%
test_df = pd.read_csv(TEST_DATA_PATH)
test_df["transliteration"] = test_df["transliteration"].apply(replace_gaps)

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

# %%
print("Starting Inference...")
all_predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
  
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            #max_length=MAX_LENGTH,
            num_beams=NUM_BEAMS,
            max_new_tokens=MAX_NEW_TOKENS,
            length_penalty=LENGTH_PENALTY,
            early_stopping=EARLY_STOPPING,
        )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_predictions.extend([d.strip() for d in decoded])

# %% [markdown]
# ## Post Processing with LLM

# %%
# =========================
# Post-processing config
# =========================
USE_OA_LEXICON = True
OA_LEXICON_PATH = "/kaggle/input/deep-past-initiative-machine-translation/OA_Lexicon_eBL.csv"
OA_THRESHOLD = 0.92  # higher = safer (less aggressive)

# OA Lexicon tuning (v2: safer)
OA_USE_TRAIN_SURFACE = True
OA_TRAIN_PATH = "/kaggle/input/deep-past-initiative-machine-translation/train.csv"
OA_MIN_SURFACE_FREQ = 3    # only use spellings that appear >= this many times in train
OA_REQUIRE_PRED_CAPITAL = True  # safest: only normalize tokens starting with uppercase
OA_ALLOW_NEAR_MATCH = False     # can help (Ashur/Assur), but may hurt if too aggressive
OA_NEAR_MAX_DIST = 1            # used only when OA_ALLOW_NEAR_MATCH=True

# OA Lexicon tuning (v3+: a bit more coverage, still safe)
OA_ALLOW_LOWERCASE_IF_TARGET = True   # also fix lowercased proper names if they are in source targets
OA_LOWER_MIN_LEN = 4
OA_MIN_SURFACE_FREQ_NAME_TYPES = 2    # for explicit NE types (DN/GN/PN...), allow rarer spellings
OA_NEAR_MIN_TARGET_FREQ = 10          # near-match only for very frequent names
OA_NEAR_MIN_LEN = 5

# =========================
# Translation memory (exact match from train)
# =========================
# Very safe if duplicates exist between train/test.
# If a test transliteration EXACTLY matches a train transliteration (after replace_gaps + optional space normalization),
# we directly output the most frequent train translation for that source.
USE_TRAIN_EXACT_MATCH = True
TRAIN_MATCH_NORMALIZE_SRC = True   # collapse multiple spaces in transliteration for matching

# Near-duplicate translation memory (optional, higher risk than exact match)
# Uses char TF-IDF on transliteration; apply only when similarity is extremely high.
USE_TRAIN_NEAR_DUP = False
NEAR_DUP_SIM_THRESHOLD = 0.995
NEAR_DUP_MIN_SRC_LEN = 20


# Optional: LLM post-edit (can be slow / can hurt BLEU if it paraphrases)
USE_LLM_POLISH = False  # set True to enable Gemma post-edit

# %%
import torch
import gc
import pandas as pd
import re
import unicodedata
from collections import defaultdict
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoProcessor
from collections import Counter, defaultdict

# %%
del model
del tokenizer
del test_loader
del test_dataset

gc.collect()
torch.cuda.empty_cache()

# %%
# -------------------------
# Load OA Lexicon and build token->lexeme index
# -------------------------
if USE_OA_LEXICON:
    print(f"📚 Loading OA Lexicon: {OA_LEXICON_PATH}")
    oa = pd.read_csv(OA_LEXICON_PATH)
    print("OA Lexicon rows:", len(oa))

    SUB_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

    def norm_key_token(s: str) -> str:
        """Key for matching transliteration tokens to lexicon tokens."""
        s = "" if s is None else str(s)
        s = unicodedata.normalize("NFKC", s).translate(SUB_DIGITS).strip()
        # remove wrapping brackets/quotes
        s = re.sub(r"^[\"'“”‘’\(\)\[\]\{\}<>]+", "", s)
        s = re.sub(r"[\"'“”‘’\(\)\[\]\{\}<>]+$", "", s)
        # trim punctuation at edges
        s = s.strip(".,;:!?")
        return s.lower()

    token2lexemes = defaultdict(list)  # token_key -> [(lexeme, type), ...]

    for _, r in oa.iterrows():
        typ = "" if pd.isna(r.get("type")) else str(r["type"]).strip()
        lex = "" if pd.isna(r.get("lexeme")) else str(r["lexeme"]).strip()
        if not lex:
            continue

        for col in ["form", "norm", "Alt_lex"]:
            if col not in oa.columns:
                continue
            v = r.get(col)
            if pd.isna(v):
                continue
            for tok in str(v).split():
                k = norm_key_token(tok)
                if k:
                    token2lexemes[k].append((lex, typ))

    # de-dup lists (keep order)
    for k, v in list(token2lexemes.items()):
        seen = set()
        uniq = []
        for lex, typ in v:
            key = (lex, typ)
            if key in seen:
                continue
            seen.add(key)
            uniq.append((lex, typ))
        token2lexemes[k] = uniq

    print("OA token keys indexed:", len(token2lexemes))
else:
    token2lexemes = defaultdict(list)


# %%
# -------------------------
# Folding + heuristics
# -------------------------
_DIACRITIC_MAP = str.maketrans({
    "š": "s", "Š": "s",
    "ṣ": "s", "Ṣ": "s",
    "ṭ": "t", "Ṭ": "t",
    "ḫ": "h", "Ḫ": "h",
    "ā": "a", "Ā": "a",
    "ē": "e", "Ē": "e",
    "ī": "i", "Ī": "i",
    "ū": "u", "Ū": "u",
    "ʾ": "", "ʼ": "", "’": "", "'": "",
})

_DIACRITIC_CHARS = set([
    "š","Š","ṣ","Ṣ","ṭ","Ṭ","ḫ","Ḫ","ā","ē","ī","ū","Ā","Ē","Ī","Ū"
])

def _strip_disambig(s: str) -> str:
    """Remove trailing numeric homograph markers: Inanna2 -> Inanna"""
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"(?<=\D)\d+$", "", s)
    return s


def fold_for_match(s: str) -> str:
    """Aggressive fold for matching name variants (diacritics/digraphs)."""
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = _strip_disambig(s)
    s = s.translate(_DIACRITIC_MAP)
    s = s.lower()
    # common ASCII digraph variants
    s = s.replace("sh", "s").replace("kh", "h")
    # keep only letters (digits often disambiguators in names)
    s = re.sub(r"[^a-z]+", "", s)
    return s


def looks_like_name(lexeme: str, typ: str) -> bool:
    if not lexeme:
        return False
    t = (typ or "").strip().upper()

    # If the lexicon has explicit NE tags, prefer them
    if t in {"DN", "GN", "PN", "MN", "ON", "TN"}:
        return True

    # heuristic: lexeme contains uppercase OR Akkadian diacritics
    if any(ch.isupper() for ch in lexeme):
        return True
    if any(ch in _DIACRITIC_CHARS for ch in lexeme):
        return True

    return False

# %%
# -------------------------
# Extra safety/coverage helpers (v4)
# -------------------------

EXPLICIT_NE_TYPES = {"DN", "GN", "PN", "MN", "ON", "TN"}

def is_explicit_ne_type(typ: str) -> bool:
    t = (typ or "").strip().upper()
    return t in EXPLICIT_NE_TYPES

# stopwords to avoid accidentally uppercasing/rewriting common words when OA_ALLOW_LOWERCASE_IF_TARGET=True
EN_STOPWORDS = {
    'the','a','an','and','or','of','to','in','on','at','by','for','from','with','as','but','not','no','nor',
    'is','are','was','were','be','been','being',
    'i','you','he','she','it','we','they','me','him','her','us','them','my','your','his','their','our','its',
    'this','that','these','those','there','here',
    'who','whom','which','what','when','where','why','how',
}


# %%
# -------------------------
# Learn the *surface spelling* from train translations
# -------------------------
fold2surface = {}
fold2freq = {}


# -------------------------
# Translation memory: exact match mapping (train source -> most common train translation)
# -------------------------
train_exact_map = {}

# -------------------------
# Near-duplicate TM (char TF-IDF)
# -------------------------
near_dup_vec = None
near_dup_nn = None
near_dup_keys = None
near_dup_tgts = None

# %%
if USE_TRAIN_NEAR_DUP and train_exact_map:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors

        # Use unique sources (train_exact_map keys) to keep it small
        near_dup_keys = list(train_exact_map.keys())
        near_dup_tgts = [train_exact_map[k] for k in near_dup_keys]

        near_dup_vec = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            min_df=2,
            lowercase=True,
        )
        X_train = near_dup_vec.fit_transform(near_dup_keys)

        near_dup_nn = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute')
        near_dup_nn.fit(X_train)

        print(f"🔁 Near-dup TM ready | train_keys={len(near_dup_keys)}")

    except Exception as e:
        print('⚠️ Near-dup TM disabled due to error:', repr(e))
        near_dup_vec = None
        near_dup_nn = None
        near_dup_keys = None
        near_dup_tgts = None


# %%
def norm_src_for_match(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    if TRAIN_MATCH_NORMALIZE_SRC:
        s = re.sub(r"\s+", " ", s)
    return s

if USE_TRAIN_EXACT_MATCH:
    try:
        print(f"🧠 Building exact-match TM from train: {OA_TRAIN_PATH}")
        _tm_train = pd.read_csv(OA_TRAIN_PATH)
        if 'transliteration' not in _tm_train.columns:
            raise ValueError('train.csv missing transliteration column')
        _tm_train['transliteration'] = _tm_train['transliteration'].apply(replace_gaps)

        tgt_col = 'translation' if 'translation' in _tm_train.columns else _tm_train.columns[-1]

        from collections import Counter, defaultdict
        tmp = defaultdict(Counter)
        for src, tgt in zip(_tm_train['transliteration'].astype(str).tolist(), _tm_train[tgt_col].astype(str).tolist()):
            k = norm_src_for_match(src)
            if not k:
                continue
            tmp[k][tgt] += 1
        # choose most frequent translation for duplicated sources
        train_exact_map = {k: c.most_common(1)[0][0] for k, c in tmp.items()}
        print('TM entries:', len(train_exact_map))
    except Exception as e:
        print('⚠️ TM build failed:', repr(e))
        train_exact_map = {}


# %%
if USE_OA_LEXICON and OA_USE_TRAIN_SURFACE:
    try:
        train_df = pd.read_csv(OA_TRAIN_PATH)
        if "translation" in train_df.columns:
            col = "translation"
        else:
            # fallback: last column
            col = train_df.columns[-1]

        surf_counter = defaultdict(Counter)

        token_re = re.compile(r"[A-Za-zšṣṭḫāēīūŠṢṬḪĀĒĪŪ'’\-]+")

        for text in train_df[col].astype(str).tolist():
            for tok in token_re.findall(text):
                if len(tok) < 3:
                    continue

                # focus on tokens that look like proper nouns in English references
                if not (tok[:1].isupper() or any(ch in _DIACRITIC_CHARS for ch in tok)):
                    continue

                f = fold_for_match(tok)
                if len(f) < 4:
                    continue

                surf_counter[f][tok] += 1

        for f, counter in surf_counter.items():
            tok, cnt = counter.most_common(1)[0]
            fold2surface[f] = tok
            fold2freq[f] = cnt

        print(f"🔎 Learned surface forms from train: {len(fold2surface)} folds")

    except Exception as e:
        print("⚠️ Could not build train surface table:", repr(e))
        fold2surface = {}
        fold2freq = {}

# %%
# -------------------------
# Optional: near match (edit distance <= 1), OFF by default
# -------------------------

def _levenshtein_leq(a: str, b: str, max_dist: int = 1) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > max_dist:
        return False

    # DP with early stop (max_dist small)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        min_cur = cur[0]
        for j, cb in enumerate(b, 1):
            ins = cur[j-1] + 1
            dele = prev[j] + 1
            sub = prev[j-1] + (ca != cb)
            v = min(ins, dele, sub)
            cur.append(v)
            if v < min_cur:
                min_cur = v
        if min_cur > max_dist:
            return False
        prev = cur
    return prev[-1] <= max_dist

def _levenshtein_distance_cap(a: str, b: str, max_dist: int = 1) -> int:
    """Return Levenshtein distance if <= max_dist, else max_dist+1 (early stop)."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        min_cur = cur[0]
        for j, cb in enumerate(b, 1):
            ins = cur[j-1] + 1
            dele = prev[j] + 1
            sub = prev[j-1] + (ca != cb)
            v = min(ins, dele, sub)
            cur.append(v)
            if v < min_cur:
                min_cur = v
        if min_cur > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]


# %%
# -------------------------
# Sentence-level target extraction + substitution
# -------------------------

def extract_name_targets(translit: str, max_targets: int = 50):
    """Return a dict: fold_key -> best_surface_token (from train)."""
    if not USE_OA_LEXICON:
        return {}

    translit = "" if translit is None else str(translit)
    targets = {}
    seen = set()

    for tok in translit.split():
        k = norm_key_token(tok)
        for lex, typ in token2lexemes.get(k, []):
            if lex in seen:
                continue
            seen.add(lex)

            if not looks_like_name(lex, typ):
                continue

            # normalize lexeme and fold
            lex_clean = _strip_disambig(lex)
            f = fold_for_match(lex_clean)
            if len(f) < 4:
                continue

            # only use spellings that appear in references (train)
            min_freq = OA_MIN_SURFACE_FREQ_NAME_TYPES if is_explicit_ne_type(typ) else OA_MIN_SURFACE_FREQ
            if fold2surface and (f in fold2surface) and (fold2freq.get(f, 0) >= min_freq):
                targets[f] = fold2surface[f]

        if len(targets) >= max_targets:
            break

    return targets

# %%
def lexicon_name_normalize(pred: str, targets: dict) -> str:
    if not pred or not targets:
        return pred

    parts = str(pred).split()
    out = []

    for p in parts:
        m = re.match(r"^(\W*)(.*?)(\W*)$", p)
        pre, core, suf = m.group(1), m.group(2), m.group(3)

        if not core:
            out.append(p)
            continue

        # Handle possessive endings: Assur's / Assur’s
        poss = ""
        core_base = core
        pm = re.match(r"^(.*?)(['’]s)$", core)
        if pm:
            core_base = pm.group(1)
            poss = pm.group(2)

        # Fold for match
        f = fold_for_match(core_base)
        if len(f) < 4:
            out.append(p)
            continue

        is_cap = core_base[:1].isupper()

        # 1) Exact folded match: apply replacement (optionally even if lowercase)
        if f in targets:
            if is_cap or (not OA_REQUIRE_PRED_CAPITAL):
                out.append(pre + targets[f] + poss + suf)
                continue

            # Allow lowercased proper names ONLY when they are in the source targets
            if OA_ALLOW_LOWERCASE_IF_TARGET and len(f) >= OA_LOWER_MIN_LEN and core_base.lower() not in EN_STOPWORDS:
                out.append(pre + targets[f] + poss + suf)
                continue

            out.append(p)
            continue

        # 2) If capital required and token isn't capitalized, do nothing
        if OA_REQUIRE_PRED_CAPITAL and not is_cap:
            out.append(p)
            continue

        # 3) Optional near match (very conservative)
        if OA_ALLOW_NEAR_MATCH and len(f) >= OA_NEAR_MIN_LEN:
            best = None
            best_dist = 999
            best_freq = -1
            for tf, surf in targets.items():
                if len(tf) < OA_NEAR_MIN_LEN:
                    continue
                # only try near-match for very frequent canonical spellings
                if fold2freq.get(tf, 0) < OA_NEAR_MIN_TARGET_FREQ:
                    continue
                if abs(len(f) - len(tf)) > OA_NEAR_MAX_DIST:
                    continue
                # quick guards
                if f[0] != tf[0] or f[-1] != tf[-1]:
                    continue

                dist = _levenshtein_distance_cap(f, tf, max_dist=OA_NEAR_MAX_DIST)
                if dist <= OA_NEAR_MAX_DIST:
                    freq = fold2freq.get(tf, 0)
                    if (dist < best_dist) or (dist == best_dist and freq > best_freq):
                        best = surf
                        best_dist = dist
                        best_freq = freq

            if best is not None:
                out.append(pre + best + poss + suf)
                continue

        out.append(p)

    return " ".join(out)

# %%
def post_process_with_oa_lexicon(translit: str, pred: str, threshold: float = None) -> str:
    # `threshold` kept for backward-compatibility (v1 notebooks), but v2 doesn't use it.
    if not USE_OA_LEXICON:
        return pred
    targets = extract_name_targets(translit)
    return lexicon_name_normalize(pred, targets)

# %%
LLM_MODEL_PATH = "/kaggle/input/gemma-3/transformers/gemma-3-4b-it/1"

# 4bit量子化でロード (T4 GPU x2環境でも動作可能に)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )

if USE_LLM_POLISH:
    print(f"🚀 Loading LLM from {LLM_MODEL_PATH}...")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        # quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
else:
    llm_tokenizer = None
    llm_model = None
    print("⚠️ USE_LLM_POLISH=False -> skip loading LLM.")

# %%
if USE_LLM_POLISH and llm_model is not None:
    print(llm_model.device)
else:
    "LLM disabled"

# %%
# 方針A: 「単語は変えない」最小編集プロンプト
def make_gemma3_prompt(draft_text: str):
    system_text = """You are a deterministic post-editor for MT outputs.
Goal: maximize exact-match metrics (BLEU/chrF). Therefore NEVER paraphrase.

ALLOWED edits (ONLY):
- whitespace normalization (remove double spaces)
- spacing around punctuation , . ; : ! ?
- normalize quotes/dashes to ASCII (' " -)
- if there is an unmatched opening '[' or '(' then ONLY add the missing closing bracket ']' or ')' at the END of the text
- capitalize the first character ONLY if it is a letter AND you do not change any other characters

FORBIDDEN:
- changing, adding, deleting, or reordering ANY words
- changing numbers
- changing proper nouns or names
- adding explanations

Output: the corrected text only (single line). If no edits needed, output the input EXACTLY."""
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": draft_text},
    ]

# %%
import re
import difflib

# ====== ルールベース正規化（LLMより安全） ======
_DASH_MAP = str.maketrans({"–": "-", "—": "-", "−": "-"})
_QUOTE_MAP = str.maketrans({"“": '"', "”": '"', "’": "'", "‘": "'"})

def basic_normalize(s: str) -> str:
    s = str(s)
    s = s.translate(_DASH_MAP).translate(_QUOTE_MAP)
    s = re.sub(r"[ \t]+", " ", s)                      # collapse spaces/tabs
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)            # no space before punctuation
    s = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", s)  # ensure a space after punctuation before letters
    s = s.strip()

    # if bracket is obviously missing, only add closers at the end (metric-safe)
    if s.count("[") > s.count("]"):
        s = s + ("]" * (s.count("[") - s.count("]")))
    if s.count("(") > s.count(")"):
        s = s + (")" * (s.count("(") - s.count(")")))

    return s

def needs_polish(s: str) -> bool:
    # 「指標に効きそうな致命傷」だけ
    if s.count("[") != s.count("]"):
        return True
    if s.count("(") != s.count(")"):
        return True
    # 句読点前スペースが多い/連発などの明確な破綻
    if re.search(r"\s+([,.;:!?])", s):
        return True
    if re.search(r"([,.;:!?])\1{1,}", s):
        return True
    return False


# ====== 強い安全装置（BLEU破壊を防ぐ） ======
PRESERVE_TERMS = [
    "Seal of", "son of", "gin", "mina", "shekel",
]
SIMILARITY_MIN = 0.985   # かなり高め（= 少しでも言い換えたら落とす）
MAX_ABS_LEN_DELTA = 12   # 末尾に ] ) を足す程度は許容
MAX_NEW_TOKENS = 128

def _alpha_tokens_lower(s: str):
    return [t.lower() for t in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", s)]

def is_safe_edit(orig: str, edited: str) -> bool:
    orig = basic_normalize(orig)
    edited = basic_normalize(edited)

    # 0) empty / very short
    if len(edited) < 3:
        return False

    # 1) words must be identical (case-insensitive) to avoid BLEU drop
    if _alpha_tokens_lower(orig) != _alpha_tokens_lower(edited):
        return False

    # 2) do not change numbers
    if re.findall(r"\d+", orig) != re.findall(r"\d+", edited):
        return False

    # 3) bracket safety: allow only adding missing closers at the END
    if orig.count("[") != edited.count("["):
        return False
    if orig.count("(") != edited.count("("):
        return False
    if edited.count("]") < orig.count("]"):
        return False
    if edited.count(")") < orig.count(")"):
        return False

    # 4) preserve key terms: if existed, must remain
    for term in PRESERVE_TERMS:
        if term in orig and term not in edited:
            return False

    # 5) string similarity
    sim = difflib.SequenceMatcher(None, orig, edited).ratio()
    if sim < SIMILARITY_MIN:
        return False

    # 6) length delta guard
    if abs(len(orig) - len(edited)) > MAX_ABS_LEN_DELTA:
        return False

    return True

def clean_llm_output(s: str) -> str:
    s = str(s).strip()

    # remove common preambles
    s = re.sub(r"^(Sure|Here(?:'s| is)|Corrected(?: text)?):\s*", "", s, flags=re.IGNORECASE).strip()

    # code fences
    if "```" in s:
        s = re.sub(r"```.*?\n", "", s, flags=re.DOTALL)
        s = s.replace("```", "").strip()

    # Gemma artifact
    if "model\n" in s:
        s = s.split("model\n")[-1].strip()

    # outer quotes
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()

    # one-line
    s = " ".join(s.splitlines()).strip()
    return s


# ====== 実行 ======
original_texts = all_predictions
translits = (
    test_df["transliteration"].astype(str).tolist()
    if "transliteration" in test_df.columns else [""] * len(original_texts)
)


# Precompute near-duplicate TM matches (vectorized)
near_dup_best = None
if USE_TRAIN_NEAR_DUP and (near_dup_vec is not None) and (near_dup_nn is not None):
    try:
        test_keys = [norm_src_for_match(s) for s in translits]
        X_test = near_dup_vec.transform(test_keys)
        dists, idxs = near_dup_nn.kneighbors(X_test, n_neighbors=1)
        sims = 1.0 - dists.ravel()
        idxs = idxs.ravel()

        near_dup_best = []
        n_ok = 0
        for s, j, sim in zip(test_keys, idxs, sims):
            if (len(s) >= NEAR_DUP_MIN_SRC_LEN) and (sim >= NEAR_DUP_SIM_THRESHOLD):
                near_dup_best.append(near_dup_tgts[int(j)])
                n_ok += 1
            else:
                near_dup_best.append(None)
        print(f"🔁 Near-dup matches: {n_ok} / {len(test_keys)} (thr={NEAR_DUP_SIM_THRESHOLD})")

    except Exception as e:
        print('⚠️ Near-dup precompute failed:', repr(e))
        near_dup_best = None

polished_texts = []
cache = {}  # deterministic decode前提でキャッシュが効く

use_llm = bool(USE_LLM_POLISH) and (llm_model is not None) and (llm_tokenizer is not None)

print(f"🧹 Post-processing {len(original_texts)} sentences | OA Lexicon={USE_OA_LEXICON} | LLM={use_llm}")

for i, (src, text) in enumerate(tqdm(zip(translits, original_texts), total=len(original_texts))):
    text = str(text)

    # Translation memory exact match override (very safe if duplicates exist)
    if USE_TRAIN_EXACT_MATCH and train_exact_map:
        k = norm_src_for_match(src)
        if k in train_exact_map:
            polished_texts.append(train_exact_map[k])
            continue

    # Near-duplicate TM override (only if exact match did not trigger)
    if USE_TRAIN_NEAR_DUP and near_dup_best is not None:
        nd = near_dup_best[i]
        if nd is not None:
            polished_texts.append(nd)
            continue


    if len(text) < 5 or text == "broken text":
        polished_texts.append(text)
        continue

    # まず安全な正規化
    norm = basic_normalize(text)

    # OA Lexicon: proper noun normalization (very safe, no paraphrase)
    if USE_OA_LEXICON:
        norm = post_process_with_oa_lexicon(src, norm, threshold=OA_THRESHOLD)

    # LLM無効ならここで確定
    if not use_llm:
        polished_texts.append(norm)
        continue

    # LLM不要ならここで確定（事故率と時間を下げる）
    if not needs_polish(norm):
        polished_texts.append(norm)
        continue

    # cache
    if norm in cache:
        polished_texts.append(cache[norm])
        continue

    messages = make_gemma3_prompt(norm)

    prompt_text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = llm_tokenizer(prompt_text, return_tensors="pt").to(llm_model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
            eos_token_id=llm_tokenizer.eos_token_id,
            pad_token_id=llm_tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][input_len:]
    response = llm_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    response = clean_llm_output(response)

    # 強いゲート：危なければ norm を採用
    if not is_safe_edit(norm, response):
        response = norm
    else:
        response = basic_normalize(response)
        if USE_OA_LEXICON:
            response = post_process_with_oa_lexicon(src, response, threshold=OA_THRESHOLD)

    cache[norm] = response
    polished_texts.append(response)

# %%
submission = pd.DataFrame({
    "id": test_df["id"],
    "translation": polished_texts
})

submission["translation"] = submission["translation"].apply(lambda x: x if len(x) > 0 else "broken text")

submission.to_csv("submission.csv", index=False)
print("Submission file saved successfully!")
submission.head()

# %%


# %%



