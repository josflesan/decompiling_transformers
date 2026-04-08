import streamlit as st
from typing import Callable

def metric_card(title: str, render_fn: Callable):
    st.subheader(title)
    render_fn()
