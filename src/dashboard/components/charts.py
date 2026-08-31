import streamlit as st
import altair as alt
import pandas as pd

def line_chart(df, x, y, color=None, log_y=False):
    if log_y:
        df = df.copy()
        df[y] = df[y].clip(lower=1e-8)
    
    st.line_chart(df, x=x, y=y, color=color)

def altair_chart(df, x, y, color="type", key=None):
    if y not in df.columns:
        return
    
    log_scale = st.checkbox("Log scale", value=False, key=key)
    y_scale = alt.Scale(type="log") if log_scale else alt.Scale()

    encodings = {
        "x": alt.X(x),
        "y": alt.Y(y, scale=y_scale),
    }

    if color is not None:
        encodings["color"] = color

    chart = alt.Chart(df).mark_line().encode(**encodings).interactive()
    st.altair_chart(chart, use_container_width=True)

def altair_trial_chart(df, x, y, color="lambda_trial", key=None, title=None):
    if y not in df.columns or df.empty:
        st.caption("No data available.")
        return

    log_scale = st.checkbox("Log scale", value=False, key=key)
    y_scale = alt.Scale(type="log") if log_scale else alt.Scale()

    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(f"{x}:Q", title=x.replace("_", " ").title()),
            y=alt.Y(f"{y}:Q", scale=y_scale, title=y.replace("_", " ").title()),
            color=alt.Color(f"{color}:N", title="Lambda run"),
            tooltip=[
                alt.Tooltip(f"{color}:N", title="Run"),
                alt.Tooltip(f"{x}:Q", title=x),
                alt.Tooltip(f"{y}:Q", title=y),
            ],
        )
        .properties(height=280, title=title)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def altair_histogram(df):
    
    with st.container():
        st.subheader("Sampler Parameter Distribution")
        
        latest_params = df["sampler_params"].dropna().iloc[-1]
        hist_df = pd.DataFrame({"value": latest_params})
        
        chart = alt.Chart(hist_df).mark_bar().encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=30), title="Parameter Value"),
            y=alt.Y("count()", title="Frequency"),
            tooltip=["count()"]
        ).properties(height=300)
        
        st.altair_chart(chart, use_container_width=True)