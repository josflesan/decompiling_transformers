"""
Visualization utilities borrowed from ARENA 3.0
"""

import torch

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from transformer_lens import HookedTransformer
from plotly.subplots import make_subplots
from typing import Dict
from torch import Tensor

from mechanistic.utilities.mechinterp_dataclasses import CircuitNode

update_layout_set = {
    "xaxis_range",
    "yaxis_range",
    "hovermode",
    "xaxis_title",
    "yaxis_title",
    "colorbar",
    "colorscale",
    "coloraxis",
    "title_x",
    "bargap",
    "bargroupgap",
    "xaxis_tickformat",
    "yaxis_tickformat",
    "title_y",
    "legend_title_text",
    "xaxis_showgrid",
    "xaxis_gridwidth",
    "xaxis_gridcolor",
    "yaxis_showgrid",
    "yaxis_gridwidth",
    "yaxis_gridcolor",
    "showlegend",
    "xaxis_tickmode",
    "yaxis_tickmode",
    "margin",
    "xaxis_visible",
    "yaxis_visible",
    "bargap",
    "bargroupgap",
    "coloraxis_showscale",
    "xaxis_tickangle",
    "yaxis_scaleanchor",
    "xaxis_tickfont",
    "yaxis_tickfont",
}

update_traces_set = {"textposition"}

def reorder_list_in_plotly_way(L: list, col_wrap: int):
    """
    Helper function, because Plotly orders figures in an annoying way when there's column wrap.
    """
    L_new = []
    while len(L) > 0:
        L_new.extend(L[-col_wrap:])
        L = L[:-col_wrap]
    return L_new

def to_numpy(tensor):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (Tensor, t.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")

def plot_path_patching_per_sender(
    results: Dict[str, float],
    sender_nodes: Dict[str, CircuitNode],
    model: HookedTransformer,
    title: str="Path Patching Heatmap",
    percent: bool=True
):
    """
    Plot path patching results per sender node
    """
    
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    mat = torch.full((n_layers, n_heads + 1), float('nan'))  # +1 column for MLP nodes
    x_labels = [f"H{h}" for h in range(n_heads)] + ["MLP"]
    
    for sender_key, score in results.items():
        node = sender_nodes[sender_key]
        layer = int(sender_key.split(".")[1])
        
        if node.head_idx is not None:
            mat[layer, node.head_idx] = score
        else:
            mat[layer, n_heads] = score  # MLP goes in last column
    
    # Plot the results
    if percent:
        mat = 100 * mat
    
    fig = px.imshow(
        mat,
        labels=dict(
            x="Head / Node Index",
            y="Layer",
            color="Effect (%)"
        ),
        x=x_labels,
        y=[f"L{l}" for l in range(n_layers)],
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
        aspect='auto'
    )
    
    fig.update_layout(title=title)
    
    return fig

def plot_path_patching_aggregated(
    results: Dict[str, float],
    receiver_nodes: Dict[str, CircuitNode],
    model: HookedTransformer,
    title: str="Path Patching Heatmap",
    percent: bool=True
):
    """
    Generic heatmap for path patching results
    """
    
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    for sender, receiver_results in results.items():
        detached = {k: v.detach() if hasattr(v, "detach") else v.numpy() for k, v in receiver_results.items()}
        results[sender] = detached
    
    mat = torch.full((n_layers, n_heads + 1), float('nan'))  # +1 column for MLP nodes
    x_labels = [f"H{h}" for h in range(n_heads)] + ["MLP"]
    
    for receiver_name, node in receiver_nodes.items():
        scores = torch.tensor([
            results[s][receiver_name] for s in results
        ])
        layer = node.layer_idx
        col = node.head_idx if node.head_idx is not None else n_heads
        mat[layer, col] = torch.mean(scores).item()
    
    # Plot the results
    if percent:
        mat = 100 * mat
    
    
    fig = px.imshow(
        mat,
        labels=dict(
            x="Head / Node Index",
            y="Layer",
            color="Effect (%)"
        ),
        x=x_labels,
        y=[f"L{l}" for l in range(n_layers)],
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
        aspect='auto'
    )
    
    fig.update_layout(title=title)
    
    return fig

def line(y: Tensor | list, renderer=None, **kwargs):
    """
    Edit to this helper function, allowing it to take args in update_layout (e.g. yaxis_range).
    """
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if ("size" in kwargs_pre) or ("shape" in kwargs_pre):
        size = kwargs_pre.pop("size", None) or kwargs_pre.pop("shape", None)
        kwargs_pre["height"], kwargs_pre["width"] = size  # type: ignore
    return_fig = kwargs_pre.pop("return_fig", False)
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    if "xaxis_tickvals" in kwargs_pre:
        tickvals = kwargs_pre.pop("xaxis_tickvals")
        kwargs_post["xaxis"] = dict(
            tickmode="array",
            tickvals=kwargs_pre.get("x", np.arange(len(tickvals))),
            ticktext=tickvals,
        )
    if "hovermode" not in kwargs_post:
        kwargs_post["hovermode"] = "x unified"
    hovertext = kwargs_pre.pop("hovertext", None)
    if "use_secondary_yaxis" in kwargs_pre and kwargs_pre["use_secondary_yaxis"]:
        del kwargs_pre["use_secondary_yaxis"]
        if "labels" in kwargs_pre:
            labels: dict = kwargs_pre.pop("labels")
            kwargs_post["yaxis_title_text"] = labels.get("y1", None)
            kwargs_post["yaxis2_title_text"] = labels.get("y2", None)
            kwargs_post["xaxis_title_text"] = labels.get("x", None)
        for k in ["title", "template", "width", "height"]:
            if k in kwargs_pre:
                kwargs_post[k] = kwargs_pre.pop(k)
        fig = make_subplots(specs=[[{"secondary_y": True}]]).update_layout(**kwargs_post)
        y0 = to_numpy(y[0])
        y1 = to_numpy(y[1])
        x0, x1 = kwargs_pre.pop("x", [np.arange(len(y0)), np.arange(len(y1))])
        name0, name1 = kwargs_pre.pop("names", ["yaxis1", "yaxis2"])
        fig.add_trace(go.Scatter(y=y0, x=x0, name=name0), secondary_y=False)
        fig.add_trace(go.Scatter(y=y1, x=x1, name=name1), secondary_y=True)
    else:
        y = (
            list(map(to_numpy, y))
            if isinstance(y, list) and not (isinstance(y[0], int) or isinstance(y[0], float))
            else to_numpy(y)
        )  # type: ignore
        names = kwargs_pre.pop("names", None)
        fig = px.line(y=y, **kwargs_pre).update_layout(**kwargs_post)
        if names is not None:
            fig.for_each_trace(lambda trace: trace.update(name=names.pop(0)))
    if hovertext is not None:
        ht = fig.data[0].hovertemplate
        fig.for_each_trace(
            lambda trace: trace.update(hovertext=hovertext, hovertemplate="%{hovertext}<br>" + ht)
        )

    return fig if return_fig else fig.show(renderer=renderer)

def imshow(tensor: Tensor, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if ("size" in kwargs_pre) or ("shape" in kwargs_pre):
        size = kwargs_pre.pop("size", None) or kwargs_pre.pop("shape", None)
        kwargs_pre["height"], kwargs_pre["width"] = size  # type: ignore
    facet_labels = kwargs_pre.pop("facet_labels", None)
    border = kwargs_pre.pop("border", False)
    return_fig = kwargs_pre.pop("return_fig", False)
    text = kwargs_pre.pop("text", None)
    xaxis_tickangle = kwargs_post.pop("xaxis_tickangle", None)
    # xaxis_tickfont = kwargs_post.pop("xaxis_tickangle", None)
    static = kwargs_pre.pop("static", False)
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "color_continuous_midpoint" not in kwargs_pre:
        kwargs_pre["color_continuous_midpoint"] = 0.0
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.imshow(to_numpy(tensor), **kwargs_pre).update_layout(**kwargs_post)
    if facet_labels:
        # Weird thing where facet col wrap means labels are in wrong order
        if "facet_col_wrap" in kwargs_pre:
            facet_labels = reorder_list_in_plotly_way(facet_labels, kwargs_pre["facet_col_wrap"])
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]["text"] = label  # type: ignore
    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    if text:
        if tensor.ndim == 2:
            # if 2D, then we assume text is a list of lists of strings
            assert isinstance(text[0], list)
            assert isinstance(text[0][0], str)
            text = [text]
        else:
            # if 3D, then text is either repeated for each facet, or different
            assert isinstance(text[0], list)
            if isinstance(text[0][0], str):
                text = [text for _ in range(len(fig.data))]
        for i, _text in enumerate(text):
            fig.data[i].update(text=_text, texttemplate="%{text}", textfont={"size": 12})
    # Very hacky way of fixing the fact that updating layout with xaxis_* only applies to first facet by default
    if xaxis_tickangle is not None:
        n_facets = 1 if tensor.ndim == 2 else tensor.shape[0]
        for i in range(1, 1 + n_facets):
            xaxis_name = "xaxis" if i == 1 else f"xaxis{i}"
            fig.layout[xaxis_name]["tickangle"] = xaxis_tickangle  # type: ignore

    return fig if return_fig else fig.show(renderer=renderer, config={"staticPlot": static})