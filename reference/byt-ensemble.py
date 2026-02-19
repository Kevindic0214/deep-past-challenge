# %%
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import re
import joblib

# %% [markdown]
# ### Load Models

# %%
CONFIG = {
    "data_path": "/kaggle/input/deep-past-initiative-machine-translation/test.csv",
    "models": [
        "/kaggle/input/byt5-base-big-data2",
        "/kaggle/input/byt5-akkadian-model",
        "/kaggle/input/train-gap-all-2/byt5-base-akkadian_gap_setence2"
    ],
    "model_weights": [0.995, 0.98, 0.395],
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "max_len": 512,
    "batch_size": 8,
    "gen_params": {
        "num_beams": 8,
        "max_new_tokens": 512,
        "length_penalty": 1.10,
        "early_stopping": True
    }
}

# %% [markdown]
# ### Preprocess & Post-Process

# %%
def preprocess_transliteration(text):
    if pd.isna(text): return ""
    processed_text = str(text)
    processed_text = re.sub(r'(\.{3,}|…+|……)', '<big_gap>', processed_text)
    processed_text = re.sub(r'(xx+|\s+x\s+)', '<gap>', processed_text)
    return processed_text

def postprocess_translation(text):
    if not isinstance(text, str) or not text.strip(): return ""
    
    processed_text = text.replace('ḫ', 'h').replace('Ḫ', 'H')
    sub_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    processed_text = processed_text.translate(sub_map)

    processed_text = re.sub(r'(\[x\]|\(x\)|\bx\b)', '<gap>', processed_text, flags=re.I)
    processed_text = re.sub(r'(\.{3,}|…|\[\.+\])', '<big_gap>', processed_text)
    
    processed_text = re.sub(r'<gap>\s*<gap>', ' <big_gap> ', processed_text)
    processed_text = re.sub(r'<big_gap>\s*<big_gap>', ' <big_gap> ', processed_text)

    processed_text = re.sub(r'\((fem|plur|pl|sing|singular|plural|\?|!)\.?\s*\w*\)', '', processed_text, flags=re.I)

    processed_text = processed_text.replace('<gap>', '\x00GAP\x00').replace('<big_gap>', '\x00BIG\x00')
    
    # Remove bad characters
    bad_chars = '!?()"—–<>⌈⌋⌊[]+ʾ/;'
    processed_text = processed_text.translate(str.maketrans('', '', bad_chars))

    processed_text = processed_text.replace('\x00GAP\x00', ' <gap> ').replace('\x00BIG\x00', ' <big_gap> ')

    # Handle fractions
    frac_map = {
        r'\.5\b': ' ½', r'\.25\b': ' ¼', r'\.75\b': ' ¾',
        r'\.33+\d*\b': ' ⅓', r'\.66+\d*\b': ' ⅔'
    }
    for pat, rep in frac_map.items():
        processed_text = re.sub(r'(\d+)' + pat, r'\1' + rep, processed_text)
        processed_text = re.sub(r'\b0' + pat, rep.strip(), processed_text)

    # Remove repeated words
    processed_text = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', processed_text)
    for n in range(4, 1, -1):
        pat = r'\b((?:\w+\s+){' + str(n-1) + r'}\w+)(?:\s+\1\b)+'
        processed_text = re.sub(pat, r'\1', processed_text)

    return re.sub(r'\s+', ' ', processed_text).strip().strip('-')

# %% [markdown]
# ### Weightings

# %%
def create_model_soup():
    total_score = sum(CONFIG['model_weights'])
    WEIGHTS = [w / total_score for w in CONFIG['model_weights']]
    
    # Use the second model as the base template
    template_model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG['models'][1])
    soup_sd = template_model.state_dict()
    
    model_1_sd = AutoModelForSeq2SeqLM.from_pretrained(CONFIG['models'][0]).state_dict()
    model_3_sd = AutoModelForSeq2SeqLM.from_pretrained(CONFIG['models'][2]).state_dict()
    
    for key in soup_sd:
        # Initialize with weighted value from template model (model 2)
        weighted_value = WEIGHTS[1] * soup_sd[key]
        norm_factor = WEIGHTS[1]
        
        # Add contributions from other models if key exists
        if key in model_1_sd:
            weighted_value += WEIGHTS[0] * model_1_sd[key]
            norm_factor += WEIGHTS[0]
        if key in model_3_sd:
            weighted_value += WEIGHTS[2] * model_3_sd[key]
            norm_factor += WEIGHTS[2]
            
        soup_sd[key] = weighted_value / norm_factor
        
    template_model.load_state_dict(soup_sd)
    return template_model.to(CONFIG['device']).eval().float()

# %% [markdown]
# ### Inference

# %%
class AkkadianTranslationDataset(Dataset):
    def __init__(self, dataframe):
        self.ids = dataframe['id'].tolist()
        self.texts = [
            "translate Akkadian to English: " 
            + str(t) for t in dataframe['transliteration']
        ]
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx): return self.ids[idx], self.texts[idx]

# %%
dataframe = pd.read_csv(CONFIG['data_path'])
dataframe['transliteration'] = dataframe['transliteration'].apply(preprocess_transliteration)

model = create_model_soup()
tokenizer = AutoTokenizer.from_pretrained(CONFIG['models'][1])

data_loader = DataLoader(
    AkkadianTranslationDataset(dataframe),
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=2,
    collate_fn=lambda batch: (
        [item[0] for item in batch],
        tokenizer(
            [item[1] for item in batch], 
            max_length=CONFIG['max_len'], 
            padding=True, truncation=True, 
            return_tensors="pt"
        )
    )
)

# %%
inference_results = []

with torch.inference_mode():
    for ids, inputs in data_loader:
        outputs = model.generate(
            input_ids=inputs.input_ids.to(CONFIG['device']),
            attention_mask=inputs.attention_mask.to(CONFIG['device']),
            **CONFIG['gen_params']
        )
        
        decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        cleaned_translations = [postprocess_translation(text) for text in decoded_texts]
        
        inference_results.extend(zip(ids, cleaned_translations))

# %% [markdown]
# ### Submission

# %%
submission_df = pd.DataFrame(inference_results, columns=['id', 'translation'])
submission_df.to_csv("submission.csv", index=False)
print(submission_df.head(10).to_string(index=False))


