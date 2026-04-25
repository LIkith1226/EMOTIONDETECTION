"""
data_preprocessing.py
---------------------
Loads emotion datasets from HuggingFace dair-ai/emotion dataset
Supports both "split" and "unsplit" versions, filters quality samples,
and cleans the text for accurate emotion detection.

Created by: Likith Abhiram Jaldu and Vishwanath Reddy
"""

import re
import string
import pandas as pd
from datasets import load_dataset

# Emotion labels from dair-ai/emotion dataset
EMOTION_LABELS = [
    "sadness", "joy", "love", "anger", "fear", "surprise"
]

ALL_LABEL_MAP: dict[int, str] = {i: name for i, name in enumerate(EMOTION_LABELS)}

# Drop emotion classes with fewer than this many single-label examples.
MIN_SAMPLES_PER_CLASS = 100

# Keep all emotions from dair-ai/emotion dataset
EXCLUDE_CLASSES = set()

# Expletives used as intensifiers are stripped so surrounding emotional
# vocabulary carries the signal ("fuck I love you" → "I love you" → joy).
PROFANITY_RE = re.compile(
    r"\b(f+u+c+k+(?:ing|er|ers|ed|face|wit|s)?|sh[i1]t(?:ty|ter|s|head|hole|bag)?"
    r"|a+s+s+(?:hole|holes|hat)?|d+a+m+n+(?:it|ed)?|b[i1]tch(?:es|ing|y)?"
    r"|bastard(?:s)?|crap(?:py|s)?|p[i1]ss(?:ed|ing)?)\b",
    re.IGNORECASE,
)

# Negation scope: join "not"/"don't"/etc. with the very next word so TF-IDF
# treats them as one token.  "do not care" → "do not_care", which has zero
# weight for the 'caring' class and nonzero for 'annoyance'/'disapproval'.
# This runs AFTER punctuation removal (so "don't" → "dont" already).
NEGATION_RE = re.compile(
    r"\b(not|no|never|dont|doesnt|cant|cannot|wont|wouldnt|shouldnt|isnt|"
    r"arent|wasnt|werent|hadnt|hasnt|havent|couldnt|neednt|didnt|"
    r"do not|does not|will not|would not|should not|could not|"
    r"is not|are not|was not|were not|had not|has not|have not)\s+(\w+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_emotions_from_hf() -> pd.DataFrame:
    """Download dair-ai/emotion dataset from HuggingFace.
    
    Supports both "split" and "unsplit" versions.
    Split version has more organized train/validation/test splits.
    Unsplit version contains all data in single set.
    """
    try:
        # Try loading the split version first (more complete)
        print("Attempting to load dair-ai/emotion dataset with 'split' configuration...")
        try:
            dataset_split = load_dataset("dair-ai/emotion", "split")
            frames = []
            for split in dataset_split.keys():
                if split not in ["train", "validation", "test"]:
                    continue
                df = pd.DataFrame(dataset_split[split])
                print(f"  Loaded {split}: {len(df)} samples")
                frames.append(df)
            
            if frames:
                result = pd.concat(frames, ignore_index=True)
                print(f"Total samples from split configuration: {len(result)}")
                return result
        except Exception as e:
            print(f"  Split configuration not available: {e}")
        
        # Fallback to unsplit version
        print("Loading dair-ai/emotion dataset with 'unsplit' configuration...")
        dataset_unsplit = load_dataset("dair-ai/emotion", "unsplit")
        df = pd.DataFrame(dataset_unsplit["train"])
        print(f"Total samples from unsplit configuration: {len(df)}")
        return df
        
    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        print("Attempting to load local CSV files...")
        return load_from_local_csv()


def load_from_local_csv() -> pd.DataFrame:
    """Load emotions from local CSV files (tweet_emotions copy.csv or tweet_emotions.csv)."""
    import os
    csv_files = ["tweet_emotions copy.csv", "tweet_emotions.csv"]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"Loading from {csv_file}...")
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} samples from {csv_file}")
            return df
    
    raise FileNotFoundError(
        f"No local CSV files found. Tried: {', '.join(csv_files)}\n"
        "Please ensure tweet_emotions.csv or tweet_emotions copy.csv is available."
    )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_emotions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and prepare emotion data. Handles both HuggingFace and local CSV formats.
    Drops classes with too few examples.
    """
    # Handle HuggingFace dair-ai/emotion format (has 'label' or 'labels' field)
    if "label" in df.columns:
        df = df.copy()
        df["emotion"] = df["label"].map(lambda x: EMOTION_LABELS[x] if isinstance(x, int) else str(x).lower())
        df = df[["text", "emotion"]]
    elif "labels" in df.columns:
        # Handle multi-label format (convert to single label)
        df = df.copy()
        df["emotion"] = df["labels"].apply(lambda lbls: EMOTION_LABELS[lbls[0]] if isinstance(lbls, list) and len(lbls) > 0 else "joy")
        df = df[["text", "emotion"]]
    elif "Emotion" in df.columns:
        # Handle local CSV format (tweet_emotions.csv)
        df = df.copy()
        df.columns = df.columns.str.lower()
        if "emotion" not in df.columns and "Emotion" in df.columns:
            df["emotion"] = df["Emotion"].str.lower()
        if "text" not in df.columns:
            # Find text column (could be 'Tweet', 'Content', etc.)
            text_cols = [col for col in df.columns if col in ["tweet", "content", "text"]]
            if text_cols:
                df["text"] = df[text_cols[0]]
        df = df[["text", "emotion"]]
    else:
        # Try to infer structure
        print("Warning: Using inferred column mapping")
        df = df.copy()
        # Find text column
        text_cols = [col for col in df.columns if col.lower() in ["text", "tweet", "content", "sentence"]]
        emotion_cols = [col for col in df.columns if col.lower() in ["emotion", "label", "sentiment"]]
        
        if text_cols and emotion_cols:
            df["text"] = df[text_cols[0]]
            df["emotion"] = df[emotion_cols[0]].astype(str).str.lower()
            df = df[["text", "emotion"]]
        else:
            raise ValueError(f"Could not infer columns from DataFrame. Available: {list(df.columns)}")
    
    # Filter by minimum samples
    counts = df["emotion"].value_counts()
    keep = [e for e, n in counts.items() if n >= MIN_SAMPLES_PER_CLASS and e not in EXCLUDE_CLASSES]
    
    if not keep:
        print(f"Warning: No emotions met minimum threshold. Using all emotions.")
        keep = [e for e in counts.index if e not in EXCLUDE_CLASSES]
    
    filtered = df[df["emotion"].isin(keep)].reset_index(drop=True)
    
    return filtered[["text", "emotion"]]


# ---------------------------------------------------------------------------
# Text cleaning  (applied identically at train time and inference time)
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    1. Lowercase
    2. Strip URLs
    3. Remove Reddit subreddit/user references
    4. Remove profanity used as intensifiers
    5. Remove punctuation  (contractions: "don't" → "dont")
    6. Join negation + next word  ("dont care" → "dont_care")
    7. Collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"r/\w+|u/\w+", " ", text)
    text = PROFANITY_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Must run AFTER punctuation removal so "don't" is already "dont"
    text = NEGATION_RE.sub(lambda m: m.group(1).replace(" ", "_") + "_" + m.group(2), text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def prepare_data(cache_path: str = "processed_data.csv") -> pd.DataFrame:
    import os
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path} ...")
        return pd.read_csv(cache_path)

    print("Loading emotion dataset ...")
    raw = load_emotions_from_hf()
    print(f"  Raw samples : {len(raw):,}")

    print("Filtering emotions ...")
    df = filter_emotions(raw)
    print(f"  After filter: {len(df):,}")
    print(f"  Emotions kept: {df['emotion'].nunique()}")
    print("\nEmotion distribution:")
    print(df["emotion"].value_counts().to_string())

    print("\nCleaning text ...")
    df["text"] = df["text"].apply(preprocess_text)
    df = df[df["text"].str.len() > 2].reset_index(drop=True)

    df.to_csv(cache_path, index=False)
    print(f"\nCached to {cache_path}")
    return df


if __name__ == "__main__":
    df = prepare_data()
    print("\nSample rows:")
    print(df.groupby("emotion").head(1).to_string(index=False))
