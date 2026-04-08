import streamlit as st
from pathlib import Path

st.set_page_config(layout="wide")

st.title("Decompiling Transformers Dashboard")
st.markdown("""
Reimplementation of "Discovering Interpretable Algorithms by Decompiling Transformers"
by Huang et al. (2026)

### Stages:
- **Pruning**: monitor pruning stages and metrics for different experiments
- **Primitive Search**: monitor primitive search training and results
- **RASP Conversion**: run and inspect final conversion results

Use the sidebar to navigate.
""")