import pandas as pd
import streamlit as st
from pathlib import Path

@st.cache_data(ttl=2)
def load_metrics(metrics_file: Path) -> pd.DataFrame:
    """
    Load metrics from a JSONL file into a DataFrame.
    Cached for fast refresh.
    """
    if not metrics_file.exists():
        return pd.DataFrame()

    try:
        df = pd.read_json(metrics_file, lines=True)
        if "timestamp" in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                # Pandas may auto-parse unix epochs as datetimes; keep unix seconds.
                df["timestamp"] = df["timestamp"].astype("int64") / 1e9
            else:
                df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Failed to load metrics: {e}")
        return pd.DataFrame()

def split_train_val(df: pd.DataFrame):
    """
    Split dataframe into train/val subsets.
    """
    if "split" not in df.columns:
        return df, pd.DataFrame()

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    
    return train_df, val_df

def get_stages(df: pd.DataFrame):
    """
    Extract and organize stages.
    """
    
    if "stage" not in df.columns:
        return []
    
    pretrain = sorted([s for s in df["stage"].unique() if "Pretrain" in s])
    prune = sorted([s for s in df["stage"].unique() if "Pretrain" not in s])
    
    return ["Overview"] + pretrain + prune