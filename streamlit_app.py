import os
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from embedding_visualizer import IntegratedPineconeVisualizer


load_dotenv()
st.set_page_config(
    page_title="Embeddings Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)


def truncate(text: Any, length: int = 100) -> str:
    if text is None:
        return ""
    s = str(text)
    return (s[:length] + "…") if len(s) > length else s


@st.cache_resource(show_spinner=False)
def get_visualizer(api_key: str) -> IntegratedPineconeVisualizer:
    return IntegratedPineconeVisualizer(api_key=api_key)


@st.cache_data(show_spinner=True)
def fetch_and_prepare(_viz: IntegratedPineconeVisualizer, index_name: str, metadata_filter: dict | None):
    vectors = _viz.fetch_vectors_with_metadata(
        index_name=index_name,
        metadata_filter=metadata_filter,
        top_k=1000,
        include_values=True,
        include_metadata=True,
    )
    if not vectors:
        return [], None
    _viz.process_vectors_to_dataframes(vectors)
    return vectors, _viz.combined_df.copy()


def sidebar_controls(df: pd.DataFrame):
    st.sidebar.header("Controls")

    method = st.sidebar.radio("Method", options=["pca", "tsne", "umap"], index=0, horizontal=True, key="method_radio")
    camera_preset = st.sidebar.radio("Camera", options=["Iso", "Front", "Top", "Reset"], index=0, horizontal=True, key="camera_radio")
    regions = st.sidebar.checkbox("Show cluster regions", value=False)
    size = st.sidebar.slider("Point size", 4, 16, 8, 1)
    opacity = st.sidebar.slider("Opacity", 0.2, 1.0, 0.85, 0.05)
    dark = st.sidebar.toggle("Dark mode", value=False)

    search = st.sidebar.text_input("Search claim_text")

    groups_selected: List[str] = []
    if df is not None and "group_id" in df.columns:
        # Build labels using group_description where available
        options = []
        counts = df["group_id"].value_counts().to_dict()
        desc_map: Dict[str, str] = {}
        if "group_description" in df.columns:
            desc_map = df.groupby("group_id")["group_description"].first().to_dict()
        for gid, cnt in counts.items():
            label = truncate(desc_map.get(gid), 100) if gid in desc_map else str(gid)
            options.append((f"{label} ({cnt})", gid))
        labels = [label for label, _ in options]
        values = [gid for _, gid in options]
        pick = st.sidebar.multiselect("Filter groups", labels, default=[], key="groups_multi")
        label_to_gid = dict(options)
        groups_selected = [label_to_gid[p] for p in pick]

    return method, camera_preset, regions, size, opacity, dark, search, groups_selected


def render_plot(viz: IntegratedPineconeVisualizer, df: pd.DataFrame, method: str, camera_preset: str, regions: bool, size: int, opacity: float, dark: bool):
    # Temporarily swap combined_df for plotting
    original = viz.combined_df
    viz.combined_df = df

    # Stable colors
    color_map = viz.build_color_map(df.get("group_id")) if df is not None else None

    fig = viz.create_3d_plot(
        method=method,
        color_by="group_id",
        save_html=None,
        show_group_labels=False,
        show_cluster_regions=regions,
        cluster_opacity=0.12,
        cluster_alphahull=8,
        display=False,
        color_map=color_map,
    )

    fig.update_traces(selector=dict(type="scatter3d"), marker=dict(size=size, opacity=opacity, line=dict(color="#ffffff", width=1.2)))
    fig.update_layout(updatemenus=[], uirevision="static")

    # Camera presets
    cams = {
        "Front": dict(eye=dict(x=2.0, y=0.0, z=0.0)),
        "Iso": dict(eye=dict(x=1.6, y=1.6, z=1.4)),
        "Top": dict(eye=dict(x=0.0, y=0.0, z=2.2)),
        "Reset": dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    }
    cam = cams.get(camera_preset or "Iso")
    if cam:
        fig.update_layout(scene_camera=cam)

    if dark:
        fig.update_layout(
            paper_bgcolor="#0b1220",
            font=dict(color="#e5e7eb"),
            scene=dict(
                bgcolor="#0b1220",
                xaxis=dict(showbackground=False, gridcolor="#23314f", showspikes=False),
                yaxis=dict(showbackground=False, gridcolor="#23314f", showspikes=False),
                zaxis=dict(showbackground=False, gridcolor="#23314f", showspikes=False),
            ),
            hoverlabel=dict(bgcolor="#111827", font_color="#f9fafb", bordercolor="#0b1220"),
        )

    viz.combined_df = original
    return fig


st.title("Embeddings Explorer")
with st.sidebar:
    st.subheader("Connection")
    default_index = os.getenv("PINECONE_INDEX", "content-gen-claim-index")
    default_key = os.getenv("PINECONE_API_KEY", "")
    index_name = st.text_input("Index", value=default_index)
    api_key = st.text_input("PINECONE_API_KEY", value=default_key, type="password")
    auto = st.checkbox("Auto-load on start (uses .env if provided)", value=bool(default_key))
    load_clicked = st.button("Load vectors", type="primary")


# Load data
viz = None
combined_df: pd.DataFrame | None = None
if api_key:
    viz = get_visualizer(api_key)
    if load_clicked or auto:
        # Metadata filter input (JSON) only shown at load time to avoid duplicate widgets
        st.sidebar.subheader("Metadata filter (JSON)")
        default_filter = "{}"
        filter_text = st.sidebar.text_area("Enter Pinecone filter JSON", value=default_filter, height=120, key="md_filter_text")
        md_filter = None
        if filter_text.strip():
            import json
            try:
                md_filter = json.loads(filter_text)
            except Exception:
                st.sidebar.warning("Invalid JSON; ignoring filter")
                md_filter = None
        with st.spinner("Fetching vectors from Pinecone…"):
            vectors, df_loaded = fetch_and_prepare(viz, index_name, md_filter)
        if not vectors:
            st.warning("No vectors found. Check index name and filters.")
        else:
            combined_df = df_loaded
else:
    st.info("Provide your PINECONE_API_KEY in the sidebar to begin.")


if viz is not None and combined_df is not None and not combined_df.empty:
    # Summary chips
    total_pts = len(combined_df)
    total_groups = combined_df["group_id"].nunique() if "group_id" in combined_df.columns else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Points", f"{total_pts}")
    c2.metric("Groups", f"{total_groups}")
    method, camera_preset, regions, size, opacity, dark, search, groups_selected = sidebar_controls(combined_df)

    # Filter
    filtered = combined_df
    if groups_selected:
        filtered = filtered[filtered.get("group_id").isin(groups_selected)]
    if search:
        try:
            filtered = filtered[filtered.get("claim_text", "").str.contains(search, case=False, na=False)]
        except Exception:
            pass

    fig = render_plot(viz, filtered, method, camera_preset, regions, size, opacity, dark)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    # Export current figure
    with st.expander("Export"):
        import plotly.io as pio
        html_str = pio.to_html(fig, full_html=True, include_plotlyjs="cdn")
        st.download_button("Download interactive HTML", data=html_str, file_name="embeddings_viz.html", mime="text/html")
else:
    st.empty()


