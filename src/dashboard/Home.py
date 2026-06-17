import streamlit as st
from pathlib import Path

st.set_page_config(layout="wide")

st.title("Decompiling Transformers Dashboard")
st.markdown("""
Reimplementation of "Discovering Interpretable Algorithms by Decompiling Transformers"
by Huang et al. (2026)

### Stages:
- **Pruning**: monitor pruning stages and metrics for different experiments
- **MLP Primitives**: monitor MLP primitive replacement progress and results
- **Attention Primitives**: monitor attention primitive replacement progress and results
- **RASP Conversion**: inspect the final symbolic D-RASP program and decompilation artifacts
- **Mechanistic**: run attribution and other mechanistic experiments for a chosen checkpoint

Use the sidebar to navigate.
""")