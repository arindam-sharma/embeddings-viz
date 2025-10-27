import os
from dotenv import load_dotenv
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import plotly.graph_objects as go
import plotly.express as px

from embedding_visualizer import IntegratedPineconeVisualizer


load_dotenv()


def build_app() -> Dash:
    app = Dash(__name__)

    # Initial configuration
    default_index = os.getenv("PINECONE_INDEX", "content-gen-claim-index")
    default_api_key = os.getenv("PINECONE_API_KEY", "")

    # Create the visualizer lazily after API key is provided
    viz_holder = {"viz": None}

    # Optionally prefetch data if API key is available in .env
    initial_options = []
    initial_legend = []
    initial_summary = ""
    initial_fig = go.Figure()
    if default_api_key:
        try:
            viz_holder["viz"] = IntegratedPineconeVisualizer(api_key=default_api_key)
            vectors = viz_holder["viz"].fetch_vectors_with_metadata(
                index_name=default_index,
                metadata_filter=None,
                top_k=1000,
                include_values=True,
                include_metadata=True,
            )
                if vectors:
                viz_holder["viz"].process_vectors_to_dataframes(vectors)
                df0 = viz_holder["viz"].combined_df
                color_field0 = viz_holder["viz"].field_config.get('color_field', 'group_id')
                if df0 is not None and color_field0 and color_field0 in df0.columns:
                    counts0 = df0[color_field0].value_counts().to_dict()
                    initial_options = [
                        {"label": f"{gid} ({cnt})", "value": gid} for gid, cnt in counts0.items()
                    ]
                    color_map0 = viz_holder["viz"].build_color_map(df0.get(color_field0))
                    # Legend with label_field if present
                    desc_map0 = {}
                    label_field0 = viz_holder["viz"].field_config.get('label_field')
                    if label_field0 and label_field0 in df0.columns:
                        desc_map0 = df0.groupby(color_field0)[label_field0].first().to_dict()
                    for gid, cnt in counts0.items():
                        desc = desc_map0.get(gid)
                        label = (str(desc)[:100] + "…") if desc and len(str(desc)) > 100 else (str(desc) if desc else str(gid))
                        initial_legend.append(
                            html.Div(
                                [
                                    html.Span(style={"display": "inline-block", "width": "12px", "height": "12px", "borderRadius": "50%", "background": color_map0.get(gid, "#999"), "marginRight": "8px"}),
                                    html.Span(label),
                                    html.Span(f"  ({cnt})", style={"color": "#6b7280", "marginLeft": "6px"}),
                                ],
                                style={"marginBottom": "6px"},
                            )
                        )
                    # Initial figure
                    initial_fig = viz_holder["viz"].create_3d_plot(
                        method="pca",
                        color_by=color_field0,
                        show_group_labels=False,
                        show_cluster_regions=False,
                        display=False,
                        color_map=color_map0,
                    )
                    initial_fig.update_traces(
                        selector=dict(type="scatter3d"),
                        marker=dict(size=8, opacity=0.85, line=dict(color="#ffffff", width=1.2)),
                    )
                initial_summary = f"Loaded {len(vectors)} vectors · {len(initial_options)} groups"
        except Exception:
            # Silent prefetch failure; UI still usable via Load
            viz_holder["viz"] = None

    # Store reduced data across callbacks
    cache = {"method": None}

    controls = html.Div(
        [
            html.Div("Embeddings Explorer", className="title"),
            html.Div(
                [
                    dcc.Input(
                        id="index-name",
                        placeholder="Pinecone index name",
                        value=default_index,
                        style={"width": "280px"},
                    ),
                    dcc.Input(
                        id="api-key",
                        type="password",
                        placeholder="PINECONE_API_KEY (or set in .env)",
                        value=default_api_key,
                        style={"width": "280px", "marginLeft": "8px"},
                    ),
                    html.Button("Load", id="load-btn"),
                    dcc.RadioItems(
                        id="method",
                        options=[
                            {"label": "PCA", "value": "pca"},
                            {"label": "TSNE", "value": "tsne"},
                            {"label": "UMAP", "value": "umap"},
                        ],
                        value="pca",
                        inline=True,
                    ),
                    dcc.Checklist(
                        id="dark-toggle",
                        options=[{"label": "Dark", "value": "dark"}],
                        value=[],
                        inline=True,
                        style={"marginLeft": "8px"},
                    ),
                    dcc.Checklist(
                        id="regions-toggle",
                        options=[{"label": "Regions", "value": "on"}],
                        value=[],
                        inline=True,
                    ),
                    dcc.Input(
                        id="search",
                        placeholder="Search claim_text...",
                        style={"width": "280px"},
                    ),
                    html.Button("Front", id="cam-front", n_clicks=0, style={"marginLeft": "8px"}),
                    html.Button("Iso", id="cam-iso", n_clicks=0),
                    html.Button("Top", id="cam-top", n_clicks=0),
                    html.Button("Reset", id="cam-reset", n_clicks=0),
                    html.Button("Export HTML", id="export-html", n_clicks=0, style={"marginLeft": "8px"}),
                    dcc.Download(id="download-html"),
                ],
                className="toolbar",
            ),
            html.Div(id="summary", className="summary"),
            html.Div(
                [
                    dcc.Dropdown(id="group-filter", multi=True, placeholder="Filter group_id"),
                    dcc.Slider(id="size", min=4, max=14, step=1, value=8),
                    dcc.Slider(id="opacity", min=0.2, max=1.0, step=0.05, value=0.85),
                ],
                className="sidebar",
            ),
        ],
        className="controls",
    )

    app.layout = html.Div(
        [
            controls,
            dcc.Loading(
                dcc.Graph(id="viz", figure=initial_fig, style={"height": "88vh"}), type="dot"
            ),
            dcc.Store(id="data-loaded", data=bool(initial_options)),
        ],
        style={"padding": "16px", "fontFamily": "Inter, system-ui, -apple-system"},
    )

    @app.callback(
        Output("data-loaded", "data"),
        Output("group-filter", "options"),
        Output("summary", "children"),
        Input("load-btn", "n_clicks"),
        State("index-name", "value"),
        State("api-key", "value"),
    )
    def load_data(_: int, index_name: str, api_key_val: str):
        api_key_to_use = api_key_val or default_api_key
        if not api_key_to_use:
            return False, [], "Set PINECONE_API_KEY in .env or the input field."

        # Initialize visualizer lazily
        viz_holder["viz"] = IntegratedPineconeVisualizer(api_key=api_key_to_use)

        vectors = viz_holder["viz"].fetch_vectors_with_metadata(
            index_name=index_name,
            metadata_filter=None,
            top_k=1000,
            include_values=True,
            include_metadata=True,
        )
        if not vectors:
            return False, [], "No vectors found."
        viz_holder["viz"].process_vectors_to_dataframes(vectors)
        groups = []
        color_field = viz_holder["viz"].field_config.get('color_field')
        if viz_holder["viz"].combined_df is not None and color_field and color_field in viz_holder["viz"].combined_df.columns:
            counts = viz_holder["viz"].combined_df[color_field].value_counts().to_dict()
            groups = [
                {"label": f"{gid} ({cnt})", "value": gid} for gid, cnt in counts.items()
            ]
        summary = f"Loaded {len(vectors)} vectors · {len(groups)} groups"
        return True, groups, summary

    @app.callback(
        Output("viz", "figure"),
        Input("data-loaded", "data"),
        Input("method", "value"),
        Input("group-filter", "value"),
        Input("search", "value"),
        Input("regions-toggle", "value"),
        Input("dark-toggle", "value"),
        Input("size", "value"),
        Input("opacity", "value"),
        Input("cam-front", "n_clicks"),
        Input("cam-iso", "n_clicks"),
        Input("cam-top", "n_clicks"),
        Input("cam-reset", "n_clicks"),
    )
    def update_plot(loaded, method, groups_selected, search_text, regions, dark_vals, size, opacity, n_front, n_iso, n_top, n_reset):
        viz = viz_holder.get("viz")
        if not loaded or viz is None or viz.combined_df is None:
            return go.Figure()

        # Get configured fields
        color_field = viz.field_config.get('color_field')
        hover_field = viz.field_config.get('hover_field')
        
        # Filter
        df = viz.combined_df.copy()
        if groups_selected and color_field and color_field in df.columns:
            df = df[df[color_field].isin(groups_selected)]
        if search_text and hover_field and hover_field in df.columns:
            try:
                df = df[df[hover_field].astype(str).str.contains(search_text, case=False, na=False)]
            except Exception:
                pass

        # Temporarily swap combined_df for plotting
        original_df = viz.combined_df
        viz.combined_df = df

        show_regions = "on" in (regions or [])

        fig = viz.create_3d_plot(
            method=method,
            color_by=color_field,
            size_by=None,
            title=None,
            hover_data=None,
            save_html=None,
            show_group_labels=False,
            show_cluster_regions=show_regions,
            cluster_opacity=0.12,
            cluster_alphahull=8,
            display=False,
        )

        # Apply size and opacity styling
        fig.update_traces(
            selector=dict(type="scatter3d"),
            marker=dict(size=size, opacity=opacity, line=dict(color="#ffffff", width=1.2)),
        )

        # Remove internal updatemenus (we manage regions in the UI)
        fig.update_layout(updatemenus=[])

        # Camera presets
        ctx = callback_context
        cam_front = dict(eye=dict(x=1.8, y=0.0, z=0.0))
        cam_iso = dict(eye=dict(x=1.6, y=1.6, z=1.4))
        cam_top = dict(eye=dict(x=0.0, y=0.0, z=2.2))
        cam_reset = dict(eye=dict(x=1.5, y=1.5, z=1.5))
        if ctx.triggered:
            trig = ctx.triggered[0]["prop_id"].split(".")[0]
            if trig == "cam-front":
                fig.update_layout(scene_camera=cam_front)
            elif trig == "cam-iso":
                fig.update_layout(scene_camera=cam_iso)
            elif trig == "cam-top":
                fig.update_layout(scene_camera=cam_top)
            elif trig == "cam-reset":
                fig.update_layout(scene_camera=cam_reset)

        # Dark mode theme override
        is_dark = "dark" in (dark_vals or [])
        if is_dark:
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
            fig.update_traces(selector=dict(type="scatter3d"), marker=dict(line=dict(color="#0b1220")))
        else:
            fig.update_layout(
                paper_bgcolor='rgba(245, 247, 250, 1)'
            )

        viz.combined_df = original_df
        return fig

    # Export HTML
    @app.callback(
        Output("download-html", "data"),
        Input("export-html", "n_clicks"),
        State("viz", "figure"),
        prevent_initial_call=True,
    )
    def download_html(n_clicks, fig_dict):
        import plotly.io as pio
        fig = go.Figure(fig_dict)
        html_str = pio.to_html(fig, full_html=True, include_plotlyjs="cdn")
        return dict(content=html_str, filename="embeddings_viz.html")

    return app


if __name__ == "__main__":
    app = build_app()
    app.run_server(host="0.0.0.0", port=int(os.getenv("PORT", 8090)), debug=True)


