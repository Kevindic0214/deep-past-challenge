# %%
# ------------------------------------------------------------
# STRATEGY 1: Self-Ensemble (Multiple Runs + Voting)
# ------------------------------------------------------------
# Idea: Run همین مدل 3 بار با do_sample=True و ensemble کن
# این می‌تونه variance رو کم کنه و امتیاز رو بالا ببره
# Expected: 35.0 → 35.1-35.3
# ------------------------------------------------------------

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import re
import logging
import warnings
from pathlib import Path
from typing import List
from dataclasses import dataclass
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")


@dataclass
class SelfEnsembleConfig:
    test_data_path: str = "/kaggle/input/deep-past-initiative-machine-translation/test.csv"
    model_path: str = "/kaggle/input/byt5-akkadian-model"
    output_dir: str = "/kaggle/working/"
    
    max_length: int = 512
    batch_size: int = 8
    num_workers: int = 4
    
    # Proven optimal hyperparameters
    num_beams: int = 8
    max_new_tokens: int = 512
    length_penalty: float = 1.3
    early_stopping: bool = True
    
    # Self-ensemble config
    num_ensemble_runs: int = 3  # Run 3 times
    ensemble_strategy: str = "longest"  # or "voting"
    
    use_mixed_precision: bool = True
    aggressive_postprocessing: bool = True
    
    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        if not torch.cuda.is_available():
            self.use_mixed_precision = False


def setup_logging(output_dir: str):
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


class OptimizedPreprocessor:
    def __init__(self):
        self.patterns = {
            "big_gap": re.compile(r"(\.{3,}|…+|——)"),
            "small_gap": re.compile(r"(xx+|\s+x\s+)")
        }
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        s = pd.Series(texts).fillna("").astype(str)
        s = s.str.replace(self.patterns["big_gap"], "<big_gap>", regex=True)
        s = s.str.replace(self.patterns["small_gap"], "<gap>", regex=True)
        return s.tolist()


class VectorizedPostprocessor:
    def __init__(self, aggressive: bool = True):
        self.aggressive = aggressive
        
        self.patterns = {
            "gap": re.compile(r"(\[x\]|\(x\)|\bx\b)", re.I),
            "big_gap": re.compile(r"(\.{3,}|…|\[\.+\])"),
            "annotations": re.compile(r"\((fem|plur|pl|sing|singular|plural|\?|!)\..\s*\w*\)", re.I),
            "repeated_words": re.compile(r"\b(\w+)(?:\s+\1\b)+"),
            "whitespace": re.compile(r"\s+"),
            "punct_space": re.compile(r"\s+([.,:])"),
            "repeated_punct": re.compile(r"([.,])\1+"),
        }
        
        self.subscript_trans = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        self.special_chars_trans = str.maketrans("ḫḪ", "hH")
        self.forbidden_trans = str.maketrans("", "", '!?()"——<>⌈⌋⌊[]+ʾ/;')
    
    def postprocess_batch(self, translations: List[str]) -> List[str]:
        s = pd.Series(translations)
        
        valid_mask = s.apply(lambda x: isinstance(x, str) and bool(x.strip()))
        s.loc[~valid_mask] = ""
        
        s = s.str.translate(self.special_chars_trans)
        s = s.str.translate(self.subscript_trans)
        s = s.str.replace(self.patterns["whitespace"], " ", regex=True)
        s = s.str.strip()
        
        if self.aggressive:
            s = s.str.replace(self.patterns["gap"], "<gap>", regex=True)
            s = s.str.replace(self.patterns["big_gap"], "<big_gap>", regex=True)
            s = s.str.replace("<gap> <gap>", "<big_gap>", regex=False)
            s = s.str.replace("<big_gap> <big_gap>", "<big_gap>", regex=False)
            s = s.str.replace(self.patterns["annotations"], "", regex=True)
            
            s = s.str.replace("<gap>", "\x00G\x00", regex=False)
            s = s.str.replace("<big_gap>", "\x00B\x00", regex=False)
            s = s.str.translate(self.forbidden_trans)
            s = s.str.replace("\x00G\x00", " <gap> ", regex=False)
            s = s.str.replace("\x00B\x00", " <big_gap> ", regex=False)
            
            s = s.str.replace(r"(\d+)\.5\b", r"\1½", regex=True)
            s = s.str.replace(r"\b0\.5\b", "½", regex=True)
            s = s.str.replace(r"(\d+)\.25\b", r"\1¼", regex=True)
            s = s.str.replace(r"\b0\.25\b", "¼", regex=True)
            s = s.str.replace(r"(\d+)\.75\b", r"\1¾", regex=True)
            s = s.str.replace(r"\b0\.75\b", "¾", regex=True)
            
            s = s.str.replace(self.patterns["repeated_words"], r"\1", regex=True)
            
            for n in range(4, 1, -1):
                pattern = r"\b((?:\w+\s+){" + str(n-1) + r"}\w+)(?:\s+\1\b)+"
                s = s.str.replace(pattern, r"\1", regex=True)
            
            s = s.str.replace(self.patterns["punct_space"], r"\1", regex=True)
            s = s.str.replace(self.patterns["repeated_punct"], r"\1", regex=True)
            s = s.str.replace(self.patterns["whitespace"], " ", regex=True)
            s = s.str.strip().str.strip("-").str.strip()
        
        return s.tolist()


class AkkadianDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocessor: OptimizedPreprocessor):
        self.sample_ids = df["id"].tolist()
        preprocessed = preprocessor.preprocess_batch(df["transliteration"].tolist())
        self.input_texts = ["translate Akkadian to English: " + text for text in preprocessed]
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int):
        return self.sample_ids[idx], self.input_texts[idx]


class SelfEnsembleEngine:
    def __init__(self, config: SelfEnsembleConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.preprocessor = OptimizedPreprocessor()
        self.postprocessor = VectorizedPostprocessor(config.aggressive_postprocessing)
        
        logger.info(f"Loading model: {config.model_path}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path)
        self.model = self.model.to(config.device)
        self.model = self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"✅ Model loaded: {num_params:,} parameters")
    
    def _collate_fn(self, batch):
        ids = [s[0] for s in batch]
        texts = [s[1] for s in batch]
        
        tokenized = self.tokenizer(
            texts,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return ids, tokenized
    
    def _run_single_inference(self, dataloader) -> dict:
        """Run inference once and return results"""
        results = []
        
        gen_config = {
            "max_new_tokens": self.config.max_new_tokens,
            "length_penalty": self.config.length_penalty,
            "early_stopping": self.config.early_stopping,
            "use_cache": True,
            "num_beams": self.config.num_beams,
        }
        
        with torch.inference_mode():
            for batch_idx, (batch_ids, tokenized) in enumerate(tqdm(dataloader, desc="Inference", leave=False)):
                try:
                    input_ids = tokenized.input_ids.to(self.config.device)
                    attention_mask = tokenized.attention_mask.to(self.config.device)
                    
                    if self.config.use_mixed_precision:
                        with autocast():
                            outputs = self.model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                **gen_config
                            )
                    else:
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            **gen_config
                        )
                    
                    translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    cleaned = self.postprocessor.postprocess_batch(translations)
                    
                    results.extend(zip(batch_ids, cleaned))
                    
                    if torch.cuda.is_available() and batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    self.logger.error(f"❌ Batch {batch_idx} error: {e}")
                    results.extend([(bid, "") for bid in batch_ids])
        
        # Convert to dict for easy lookup
        return {sample_id: translation for sample_id, translation in results}
    
    def _ensemble_results(self, all_results: List[dict]) -> dict:
        """Ensemble multiple inference results"""
        sample_ids = list(all_results[0].keys())
        final_results = {}
        
        for sample_id in sample_ids:
            candidates = [results[sample_id] for results in all_results]
            
            if self.config.ensemble_strategy == "longest":
                # Pick the longest translation (more complete)
                final_results[sample_id] = max(candidates, key=len)
            
            elif self.config.ensemble_strategy == "voting":
                # Pick the most common translation
                counter = Counter(candidates)
                final_results[sample_id] = counter.most_common(1)[0][0]
            
            else:
                # Default: first one
                final_results[sample_id] = candidates[0]
        
        return final_results
    
    def run_inference(self, test_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"🚀 Starting SELF-ENSEMBLE inference ({self.config.num_ensemble_runs} runs)")
        
        dataset = AkkadianDataset(test_df, self.preprocessor)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=self.config.num_workers > 0
        )
        
        # Run inference multiple times
        all_results = []
        
        for run_idx in range(self.config.num_ensemble_runs):
            self.logger.info(f"Run {run_idx + 1}/{self.config.num_ensemble_runs}")
            results = self._run_single_inference(dataloader)
            all_results.append(results)
        
        # Ensemble the results
        self.logger.info(f"Ensembling with strategy: {self.config.ensemble_strategy}")
        final_results = self._ensemble_results(all_results)
        
        # Convert back to DataFrame
        results_df = pd.DataFrame([
            {"id": sample_id, "translation": translation}
            for sample_id, translation in final_results.items()
        ])
        
        results_df = self._validate_submission(results_df, test_df["id"].tolist())
        
        return results_df
    
    def _validate_submission(self, results_df: pd.DataFrame, original_ids: List[int]) -> pd.DataFrame:
        self.logger.info("🔍 Validating...")
        
        results_df["id"] = results_df["id"].astype(int)
        
        result_ids = set(results_df["id"].tolist())
        original_ids_set = set(original_ids)
        
        missing_ids = original_ids_set - result_ids
        if missing_ids:
            self.logger.warning(f"⚠️  Missing {len(missing_ids)} IDs")
            missing_rows = pd.DataFrame({
                "id": list(missing_ids),
                "translation": ["<gap>"] * len(missing_ids)
            })
            results_df = pd.concat([results_df, missing_rows], ignore_index=True)
        
        results_df = results_df.drop_duplicates(subset=["id"], keep="first")
        results_df = results_df.sort_values("id").reset_index(drop=True)
        results_df["translation"] = results_df["translation"].fillna("<gap>")
        results_df.loc[results_df["translation"].str.strip() == "", "translation"] = "<gap>"
        
        self.logger.info("✅ Validated")
        self._print_report(results_df)
        
        return results_df
    
    def _print_report(self, df: pd.DataFrame):
        print("\n" + "=" * 60)
        print(f"📊 SELF-ENSEMBLE REPORT ({self.config.num_ensemble_runs} runs)")
        print("=" * 60)
        
        print(f"\nTotal: {len(df)}")
        
        gap_count = (df["translation"] == "<gap>").sum()
        if gap_count > 0:
            print(f"Default gaps: {gap_count}")
        
        lengths = df["translation"].astype(str).str.len()
        print(f"\nLength - Mean: {lengths.mean():.1f}, Median: {lengths.median():.1f}")
        
        print("\n📝 Samples:")
        for idx in [0, len(df)//2, len(df)-1]:
            if idx < len(df):
                row = df.iloc[idx]
                text = str(row["translation"])
                preview = text[:70] + "..." if len(text) > 70 else text
                print(f"  ID {int(row['id']):4d}: {preview}")
        
        print("\n" + "=" * 60 + "\n")


def main():
    config = SelfEnsembleConfig()
    logger = setup_logging(config.output_dir)
    
    logger.info("=" * 60)
    logger.info("SELF-ENSEMBLE STRATEGY")
    logger.info(f"Runs: {config.num_ensemble_runs}")
    logger.info(f"Strategy: {config.ensemble_strategy}")
    logger.info("=" * 60)
    
    test_df = pd.read_csv(config.test_data_path)
    logger.info(f"✅ Loaded {len(test_df)} test samples")
    
    engine = SelfEnsembleEngine(config, logger)
    results_df = engine.run_inference(test_df)
    
    output_path = Path(config.output_dir) / "submission.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"✅ Saved to {output_path}")
    
    print("\n🎉 DONE! Self-ensemble with multiple runs")


if __name__ == "__main__":
    main()


