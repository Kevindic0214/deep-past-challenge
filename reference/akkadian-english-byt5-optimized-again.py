# %% [markdown]
# ## 1️⃣ System-Level Optimization
# 
# Set optimal environment variables **before** importing PyTorch.

# %%
import os

# Optimize PyTorch/CUDA performance
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

print("✅ Environment variables optimized")

# %% [markdown]
# ## 2️⃣ Imports & Setup

# %%
import re
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import json
import random

warnings.filterwarnings('ignore')

# Check GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# %% [markdown]
# ## 3️⃣ Configuration
# 
# **📝 EDIT THESE PATHS:**

# %%
@dataclass
class UltraConfig:
    """Ultra-optimized configuration"""
    
    # ============ PATHS - EDIT THESE ============
    test_data_path: str = "/kaggle/input/deep-past-initiative-machine-translation/test.csv"
    model_path: str = "/kaggle/input/final-byt5/byt5-akkadian-optimized-34x"
    output_dir: str = "/kaggle/working/"
    
    # ============ PROCESSING ============
    max_length: int = 512
    batch_size: int = 8  # Will auto-tune if use_auto_batch_size=True
    num_workers: int = 4  # Increased for better throughput
    
    # ============ GENERATION ============
    num_beams: int = 8
    max_new_tokens: int = 512
    length_penalty: float = 1.5
    repetition_penalty: float = 1.2
    early_stopping: bool = True
    no_repeat_ngram_size: int = 0  # Set to 3 if you see repetition
    
    # ============ OPTIMIZATIONS ============
    use_mixed_precision: bool = True      # FP16 for 2x speedup
    use_better_transformer: bool = True   # 20-50% speedup
    use_bucket_batching: bool = True      # 20-40% less padding
    use_vectorized_postproc: bool = True  # 3-5x faster postproc
    use_adaptive_beams: bool = True       # Smart beam allocation
    use_auto_batch_size: bool = False     # Auto-find optimal batch size
    
    # ============ OTHER ============
    aggressive_postprocessing: bool = True
    checkpoint_freq: int = 100
    num_buckets: int = 4  # For bucket batching
    
    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        
        if not torch.cuda.is_available():
            self.use_mixed_precision = False
            self.use_better_transformer = False

# Create config
config = UltraConfig()

print("\n📋 Configuration:")
print(f"  Device: {config.device}")
print(f"  Batch size: {config.batch_size}")
print(f"  Beams: {config.num_beams}")
print(f"\n🚀 Optimizations:")
print(f"  Mixed Precision: {config.use_mixed_precision}")
print(f"  BetterTransformer: {config.use_better_transformer}")
print(f"  Bucket Batching: {config.use_bucket_batching}")
print(f"  Vectorized Postproc: {config.use_vectorized_postproc}")
print(f"  Adaptive Beams: {config.use_adaptive_beams}")

# %% [markdown]
# ## 4️⃣ Logging Setup

# %%
def setup_logging(output_dir: str = './outputs'):
    """Setup logging"""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    log_file = Path(output_dir) / 'inference_ultra.log'
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging(config.output_dir)
logger.info("Logging initialized")

# %% [markdown]
# ## 5️⃣ Optimized Text Preprocessor
# 
# Uses pre-compiled regex patterns for speed.

# %%
class OptimizedPreprocessor:
    """Preprocessor with pre-compiled patterns"""
    
    def __init__(self):
        # Pre-compile regex patterns (20-30% faster)
        self.patterns = {
            'big_gap': re.compile(r'(\.{3,}|…+|……)'),
            'small_gap': re.compile(r'(xx+|\s+x\s+)'),
        }
    
    def preprocess_input_text(self, text: str) -> str:
        """Single text preprocessing"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        text = self.patterns['big_gap'].sub('<big_gap>', text)
        text = self.patterns['small_gap'].sub('<gap>', text)
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Vectorized batch preprocessing (faster)"""
        s = pd.Series(texts).fillna("")
        s = s.astype(str)
        s = s.str.replace(self.patterns['big_gap'], '<big_gap>', regex=True)
        s = s.str.replace(self.patterns['small_gap'], '<gap>', regex=True)
        return s.tolist()

# Test
preprocessor = OptimizedPreprocessor()
test = "lugal ... xxx mu.2.kam"
print(f"Test input:  {test}")
print(f"Preprocessed: {preprocessor.preprocess_input_text(test)}")

# %% [markdown]
# ## 6️⃣ Vectorized Postprocessor
# 
# Uses pandas for batch operations → **3-5x faster** than loop-based postprocessing.

# %%
class VectorizedPostprocessor:
    """Ultra-fast vectorized postprocessing"""
    
    def __init__(self, aggressive: bool = True):
        self.aggressive = aggressive
        
        # Pre-compile ALL patterns
        self.patterns = {
            'gap': re.compile(r'(\[x\]|\(x\)|\bx\b)', re.I),
            'big_gap': re.compile(r'(\.{3,}|…|\[\.+\])'),
            'annotations': re.compile(r'\((fem|plur|pl|sing|singular|plural|\?|!)\..\s*\w*\)', re.I),
            'repeated_words': re.compile(r'\b(\w+)(?:\s+\1\b)+'),
            'whitespace': re.compile(r'\s+'),
            'punct_space': re.compile(r'\s+([.,:])'),
            'repeated_punct': re.compile(r'([.,])\1+'),
        }
        
        # Character translation tables
        self.subscript_trans = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        self.special_chars_trans = str.maketrans('ḫḪ', 'hH')
        self.forbidden_chars = '!?()"——<>⌈⌋⌊[]+ʾ/;'
        self.forbidden_trans = str.maketrans('', '', self.forbidden_chars)
    
    def postprocess_batch(self, translations: List[str]) -> List[str]:
        """Vectorized batch postprocessing - 3-5x faster than loop"""
        
        # Convert to Series for vectorized operations
        s = pd.Series(translations)
        
        # Filter invalid entries
        valid_mask = s.apply(lambda x: isinstance(x, str) and x.strip())
        if not valid_mask.all():
            s[~valid_mask] = ""
        
        # Basic cleaning (always applied)
        s = s.str.translate(self.special_chars_trans)
        s = s.str.translate(self.subscript_trans)
        s = s.str.replace(self.patterns['whitespace'], ' ', regex=True)
        s = s.str.strip()
        
        if self.aggressive:
            # Normalize gaps
            s = s.str.replace(self.patterns['gap'], '<gap>', regex=True)
            s = s.str.replace(self.patterns['big_gap'], '<big_gap>', regex=True)
            
            # Merge adjacent gaps
            s = s.str.replace('<gap> <gap>', '<big_gap>', regex=False)
            s = s.str.replace('<big_gap> <big_gap>', '<big_gap>', regex=False)
            
            # Remove annotations
            s = s.str.replace(self.patterns['annotations'], '', regex=True)
            
            # Protect gaps during char removal
            s = s.str.replace('<gap>', '\x00GAP\x00', regex=False)
            s = s.str.replace('<big_gap>', '\x00BIG\x00', regex=False)
            
            # Remove forbidden characters
            s = s.str.translate(self.forbidden_trans)
            
            # Restore gaps
            s = s.str.replace('\x00GAP\x00', ' <gap> ', regex=False)
            s = s.str.replace('\x00BIG\x00', ' <big_gap> ', regex=False)
            
            # Fractions (vectorized)
            s = s.str.replace(r'(\d+)\.5\b', r'\1½', regex=True)
            s = s.str.replace(r'\b0\.5\b', '½', regex=True)
            s = s.str.replace(r'(\d+)\.25\b', r'\1¼', regex=True)
            s = s.str.replace(r'\b0\.25\b', '¼', regex=True)
            s = s.str.replace(r'(\d+)\.75\b', r'\1¾', regex=True)
            s = s.str.replace(r'\b0\.75\b', '¾', regex=True)
            
            # Remove repeated words
            s = s.str.replace(self.patterns['repeated_words'], r'\1', regex=True)
            
            # Remove repeated n-grams
            for n in range(4, 1, -1):
                pattern = r'\b((?:\w+\s+){' + str(n-1) + r'}\w+)(?:\s+\1\b)+'
                s = s.str.replace(pattern, r'\1', regex=True)
            
            # Fix punctuation
            s = s.str.replace(self.patterns['punct_space'], r'\1', regex=True)
            s = s.str.replace(self.patterns['repeated_punct'], r'\1', regex=True)
            
            # Final cleanup
            s = s.str.replace(self.patterns['whitespace'], ' ', regex=True)
            s = s.str.strip().str.strip('-').str.strip()
        
        return s.tolist()

# Test
postprocessor = VectorizedPostprocessor(aggressive=config.aggressive_postprocessing)
test_outputs = [
    "The king (plur.) took the city... [x] [x]",
    "He spoke spoke to the assembly"
]
cleaned = postprocessor.postprocess_batch(test_outputs)
print("Test postprocessing:")
for orig, clean in zip(test_outputs, cleaned):
    print(f"  {orig}")
    print(f"  → {clean}")

# %% [markdown]
# ## 7️⃣ Bucket Batch Sampler
# 
# Groups samples by length to minimize padding → **20-40% faster**.

# %%
class BucketBatchSampler(Sampler):
    """Batch samples by similar length to minimize padding"""
    
    def __init__(self, dataset, batch_size: int, num_buckets: int = 4, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Calculate lengths
        lengths = [len(text.split()) for _, text in dataset]
        
        # Sort indices by length
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        
        # Create buckets
        bucket_size = len(sorted_indices) // num_buckets
        self.buckets = []
        for i in range(num_buckets):
            start = i * bucket_size
            end = None if i == num_buckets - 1 else (i + 1) * bucket_size
            self.buckets.append(sorted_indices[start:end])
        
        # Log bucket info
        logger.info(f"Created {num_buckets} buckets:")
        for i, bucket in enumerate(self.buckets):
            bucket_lengths = [lengths[idx] for idx in bucket]
            logger.info(f"  Bucket {i}: {len(bucket)} samples, "
                       f"length range [{min(bucket_lengths)}, {max(bucket_lengths)}]")
    
    def __iter__(self):
        for bucket in self.buckets:
            if self.shuffle:
                random.shuffle(bucket)
            
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i+self.batch_size]
    
    def __len__(self):
        return sum((len(b) + self.batch_size - 1) // self.batch_size for b in self.buckets)

# %% [markdown]
# ## 8️⃣ Dataset Class

# %%
class AkkadianDataset(Dataset):
    """Optimized dataset with batch preprocessing"""
    
    def __init__(self, dataframe: pd.DataFrame, preprocessor: OptimizedPreprocessor):
        self.sample_ids = dataframe['id'].tolist()
        
        # Batch preprocess (faster than loop)
        raw_texts = dataframe['transliteration'].tolist()
        preprocessed = preprocessor.preprocess_batch(raw_texts)
        
        # Add task prefix
        self.input_texts = [
            "translate Akkadian to English: " + text
            for text in preprocessed
        ]
        
        logger.info(f"Dataset created with {len(self.sample_ids)} samples")
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, index: int):
        return self.sample_ids[index], self.input_texts[index]

# %% [markdown]
# ## 9️⃣ Ultra-Optimized Inference Engine
# 
# Main inference engine with all optimizations.

# %%
class UltraInferenceEngine:
    """Ultra-optimized inference engine"""
    
    def __init__(self, config: UltraConfig):
        self.config = config
        self.preprocessor = OptimizedPreprocessor()
        self.postprocessor = VectorizedPostprocessor(aggressive=config.aggressive_postprocessing)
        self.results = []
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load and optimize model"""
        logger.info(f"Loading model from {self.config.model_path}")
        
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_path
            ).to(self.config.device).eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            
            num_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model loaded: {num_params:,} parameters")
            
            # Apply BetterTransformer
            if self.config.use_better_transformer and torch.cuda.is_available():
                try:
                    from optimum.bettertransformer import BetterTransformer
                    logger.info("Applying BetterTransformer...")
                    self.model = BetterTransformer.transform(self.model)
                    logger.info("✅ BetterTransformer applied (20-50% speedup)")
                except ImportError:
                    logger.warning("⚠️  'optimum' not installed, skipping BetterTransformer")
                    logger.warning("   Install with: !pip install optimum")
                except Exception as e:
                    logger.warning(f"⚠️  BetterTransformer failed: {e}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _collate_fn(self, batch_samples):
        """Collate function"""
        batch_ids = [s[0] for s in batch_samples]
        batch_texts = [s[1] for s in batch_samples]
        
        tokenized = self.tokenizer(
            batch_texts,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return batch_ids, tokenized
    
    def find_optimal_batch_size(self, dataset, start_bs: int = 32):
        """Binary search for optimal batch size"""
        logger.info("🔍 Finding optimal batch size...")
        
        max_bs = start_bs
        min_bs = 1
        
        while max_bs - min_bs > 1:
            test_bs = (max_bs + min_bs) // 2
            
            try:
                test_batch = [dataset[i] for i in range(min(test_bs, len(dataset)))]
                ids, inputs = self._collate_fn(test_batch)
                
                with torch.inference_mode():
                    if self.config.use_mixed_precision:
                        with autocast():
                            outputs = self.model.generate(
                                input_ids=inputs.input_ids.to(self.config.device),
                                attention_mask=inputs.attention_mask.to(self.config.device),
                                num_beams=self.config.num_beams,
                                max_new_tokens=64,
                                use_cache=True
                            )
                    else:
                        outputs = self.model.generate(
                            input_ids=inputs.input_ids.to(self.config.device),
                            attention_mask=inputs.attention_mask.to(self.config.device),
                            num_beams=self.config.num_beams,
                            max_new_tokens=64,
                            use_cache=True
                        )
                
                min_bs = test_bs
                logger.info(f"  ✅ Batch size {test_bs} works")
                
                del outputs, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    max_bs = test_bs
                    logger.info(f"  ❌ Batch size {test_bs} OOM")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise
        
        optimal = min_bs
        logger.info(f"🎯 Optimal batch size: {optimal}")
        return optimal
    
    def _get_adaptive_beam_size(self, input_ids, attention_mask):
        """Adaptive beam size based on complexity"""
        if not self.config.use_adaptive_beams:
            return self.config.num_beams
        
        lengths = attention_mask.sum(dim=1)
        
        # Short → fewer beams, Long → more beams
        beam_sizes = torch.where(
            lengths < 100,
            torch.tensor(max(4, self.config.num_beams // 2)),
            torch.tensor(self.config.num_beams)
        )
        
        return beam_sizes[0].item()
    
    def _save_checkpoint(self):
        """Save checkpoint"""
        if len(self.results) > 0 and len(self.results) % self.config.checkpoint_freq == 0:
            path = Path(self.config.output_dir) / f"checkpoint_{len(self.results)}.csv"
            df = pd.DataFrame(self.results, columns=['id', 'translation'])
            df.to_csv(path, index=False)
            logger.info(f"💾 Checkpoint: {len(self.results)} translations")
    
    def run_inference(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Run ultra-optimized inference"""
        logger.info("🚀 Starting ULTRA-OPTIMIZED inference")
        
        # Create dataset
        dataset = AkkadianDataset(test_df, self.preprocessor)
        
        # Auto-find batch size
        if self.config.use_auto_batch_size:
            optimal_bs = self.find_optimal_batch_size(dataset)
            self.config.batch_size = optimal_bs
        
        # Create dataloader
        if self.config.use_bucket_batching:
            batch_sampler = BucketBatchSampler(
                dataset, 
                self.config.batch_size,
                num_buckets=self.config.num_buckets
            )
            dataloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=self.config.num_workers,
                collate_fn=self._collate_fn,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True if self.config.num_workers > 0 else False
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=self._collate_fn,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True if self.config.num_workers > 0 else False
            )
        
        logger.info(f"DataLoader created: {len(dataloader)} batches")
        logger.info(f"Active optimizations:")
        logger.info(f"  ✅ Mixed Precision: {self.config.use_mixed_precision}")
        logger.info(f"  ✅ BetterTransformer: {self.config.use_better_transformer}")
        logger.info(f"  ✅ Bucket Batching: {self.config.use_bucket_batching}")
        logger.info(f"  ✅ Vectorized Postproc: {self.config.use_vectorized_postproc}")
        logger.info(f"  ✅ Adaptive Beams: {self.config.use_adaptive_beams}")
        
        # Generation config
        # Generation config
        base_gen_config = {
            "max_new_tokens": self.config.max_new_tokens,
            "length_penalty": self.config.length_penalty,
            "repetition_penalty": self.config.repetition_penalty,  # ADD THIS LINE
            "early_stopping": self.config.early_stopping,
            "use_cache": True,
        }
        if self.config.no_repeat_ngram_size > 0:
            base_gen_config["no_repeat_ngram_size"] = self.config.no_repeat_ngram_size
        
        # Run inference
        self.results = []
        
        with torch.inference_mode():
            for batch_idx, (batch_ids, tokenized) in enumerate(tqdm(dataloader, desc="🚀 Translating")):
                try:
                    input_ids = tokenized.input_ids.to(self.config.device)
                    attention_mask = tokenized.attention_mask.to(self.config.device)
                    
                    # Adaptive beam size
                    beam_size = self._get_adaptive_beam_size(input_ids, attention_mask)
                    gen_config = {**base_gen_config, "num_beams": beam_size}
                    
                    # Generate
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
                    
                    # Decode
                    translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    
                    # Postprocess (vectorized)
                    if self.config.use_vectorized_postproc:
                        cleaned = self.postprocessor.postprocess_batch(translations)
                    else:
                        # Fallback to single processing
                        cleaned = [self.postprocessor.postprocess_batch([t])[0] for t in translations]
                    
                    # Store
                    self.results.extend(zip(batch_ids, cleaned))
                    
                    # Checkpoint
                    self._save_checkpoint()
                    
                    # Memory cleanup
                    if torch.cuda.is_available() and batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"❌ Batch {batch_idx} error: {e}")
                    self.results.extend([(bid, "") for bid in batch_ids])
                    continue
        
        logger.info("✅ Inference completed")
        
        # Create results
        results_df = pd.DataFrame(self.results, columns=['id', 'translation'])
        self._validate_results(results_df)
        
        return results_df
    
    def _validate_results(self, df: pd.DataFrame):
        """Validation report"""
        print("\n" + "="*60)
        print("📊 VALIDATION REPORT")
        print("="*60)
        
        empty = df['translation'].str.strip().eq('').sum()
        print(f"\nEmpty: {empty} ({empty/len(df)*100:.2f}%)")
        
        lengths = df['translation'].str.len()
        print(f"\n📏 Length stats:")
        print(f"   Mean: {lengths.mean():.1f}, Median: {lengths.median():.1f}")
        print(f"   Min: {lengths.min()}, Max: {lengths.max()}")
        
        short = ((lengths < 5) & (lengths > 0)).sum()
        if short > 0:
            print(f"   ⚠️  {short} very short translations")
        
        print(f"\n📝 Sample translations:")
        for idx in [0, len(df)//2, -1]:
            s = df.iloc[idx]
            preview = s['translation'][:70] + "..." if len(s['translation']) > 70 else s['translation']
            print(f"   ID {s['id']:4d}: {preview}")
        
        print("\n" + "="*60 + "\n")

print("✅ Inference engine defined")

# %% [markdown]
# ## 🔟 Load Test Data

# %%
logger.info(f"Loading test data from {config.test_data_path}")

test_df = pd.read_csv(config.test_data_path, encoding='utf-8')
logger.info(f"✅ Loaded {len(test_df)} test samples")

print("\nFirst 5 samples:")
print(test_df.head())

# %% [markdown]
# ## 1️⃣1️⃣ Run Ultra-Optimized Inference
# 
# **This is the main cell - all optimizations are active!**

# %%
# Create engine
engine = UltraInferenceEngine(config)

# Run inference
results_df = engine.run_inference(test_df)

# %% [markdown]
# ## 1️⃣2️⃣ Save Results

# %%
# Save submission
output_path = Path(config.output_dir) / 'submission.csv'
results_df.to_csv(output_path, index=False)
logger.info(f"\n✅ Submission saved to {output_path}")

# Save config
config_dict = {
    "batch_size": config.batch_size,
    "num_beams": config.num_beams,
    "length_penalty": config.length_penalty,
    "no_repeat_ngram_size": config.no_repeat_ngram_size,
    "optimizations": {
        "mixed_precision": config.use_mixed_precision,
        "better_transformer": config.use_better_transformer,
        "bucket_batching": config.use_bucket_batching,
        "vectorized_postproc": config.use_vectorized_postproc,
        "adaptive_beams": config.use_adaptive_beams,
    }
}

config_path = Path(config.output_dir) / 'ultra_config.json'
with open(config_path, 'w') as f:
    json.dump(config_dict, f, indent=2)

print("\n" + "="*60)
print("🎉 ULTRA-OPTIMIZED INFERENCE COMPLETE!")
print("="*60)
print(f"Submission file: {output_path}")
print(f"Config file: {config_path}")
print(f"Log file: {Path(config.output_dir) / 'inference_ultra.log'}")
print(f"Total translations: {len(results_df)}")
print("="*60)

# %% [markdown]
# ## 1️⃣3️⃣ [Optional] Inspect Results

# %%
# Load submission
submission = pd.read_csv(output_path)

print(f"Submission shape: {submission.shape}")
print(f"\nFirst 10 translations:")
print(submission.head(10))

print(f"\nLast 10 translations:")
print(submission.tail(10))

# Statistics
lengths = submission['translation'].str.len()
print(f"\nLength distribution:")
print(lengths.describe())

# Check for issues
empty = submission['translation'].str.strip().eq('').sum()
print(f"\nEmpty translations: {empty}")

if empty > 0:
    print("\nEmpty translation IDs:")
    print(submission[submission['translation'].str.strip().eq('')]['id'].tolist())


