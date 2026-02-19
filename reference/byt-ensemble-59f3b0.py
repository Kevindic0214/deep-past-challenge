# %% [markdown]
# This notebook demonstrates how to ensemble two fine-tuned ByT5 models to improve translation performance.
# 
# Ensembling averages the probabilities (logits) from multiple models, which often leads to more robust predictions.

# %% [markdown]
# ## 1. Imports and Setup
# Import necessary libraries and set up the environment.

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
# ## 2. Model Configuration
# Define paths to the two fine-tuned ByT5 models.

# %%
MODEL1_PATH = "/kaggle/input/final-byt5"
MODEL2_PATH = "/kaggle/input/byt5-akkadian-model"

# %%
def replace_gaps(text):
    if pd.isna(text): 
        return text
    
    text = re.sub(r'\.3(?:\s+\.3)+\.{3}(?:\s+\.{3})+\s+\.{3}(?:\s+\.{3})+', '<big_gap>', text)
    text = re.sub(r'\.3(?:\s+\.3)+\.{3}(?:\s+\.{3})+', '<big_gap>', text)
    text = re.sub(r'\.{3}(?:\s+\.{3})+', '<big_gap>', text)

    text = re.sub(r'xx', '<gap>', text)
    text = re.sub(r' x ', ' <gap> ', text)
    text = re.sub(r'……', '<big_gap>', text)
    text = re.sub(r'\.\.\.\.\.\.', '<big_gap>', text)
    text = re.sub(r'…', '<big_gap>', text)
    text = re.sub(r'\.\.\.', '<big_gap>', text)

    return text

def replace_gaps_back(text):
    if pd.isna(text):  
        return text
    
    text = re.sub(r'<gap>', 'x', text)
    text = re.sub(r'<big_gap>', '...', text)

    return text

# %%
TEST_DATA_PATH = "/kaggle/input/deep-past-initiative-machine-translation/test.csv"
BATCH_SIZE = 16
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Weighted Ensemble
# Model 1 has 0.94x performance of Model 2
w1 = 0.94 / 1.94
w2 = 1.00 / 1.94

print("Loading Model 1...")
model1 = AutoModelForSeq2SeqLM.from_pretrained(MODEL1_PATH)
sd1 = model1.state_dict()

print("Loading Model 2...")
model2 = AutoModelForSeq2SeqLM.from_pretrained(MODEL2_PATH)
sd2 = model2.state_dict()

print("Averaging weights...")
for key in sd1:
    sd2[key] = w1 * sd1[key] + w2 * sd2[key]

model = model2
model.load_state_dict(sd2)
model = model.to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL2_PATH)

test_df = pd.read_csv(TEST_DATA_PATH)
test_df['transliteration'] = test_df['transliteration'].apply(replace_gaps)

# %% [markdown]
# ## 3. Dataset Class
# Custom Dataset class to handle Akkadian transliteration inputs.

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

print("Starting Inference...")
all_predictions = []

# %% [markdown]
# ## 4. Ensemble Inference
# Generate translations by averaging the logits from both models.
# This function handles tokenization, inference, and decoding.

# %%
import torch
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(DEVICE)
model.eval()
model.float()

all_predictions = []

torch.set_grad_enabled(False)

with torch.inference_mode():
    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(DEVICE)              # int64, normal
        attention_mask = batch["attention_mask"].to(DEVICE)    # int64/bool, normal

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True,
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_predictions.extend([d.strip() for d in decoded])
        
        predictions = []
        for txt in all_predictions:
            predictions.append(txt)

# %%
print(next(model.parameters()).dtype)

# %% [markdown]
# ## 5. Submission
# Format the results and save them to a CSV file for submission.

# %%
submission = pd.DataFrame({
    "id": test_df["id"],
    "translation": predictions
})

submission["translation"] = submission["translation"].apply(lambda x: x if len(x) > 0 else "broken text")

    
submission.to_csv("submission.csv", index=False)
print("Submission file saved successfully!")
submission.head()


