import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Any, Optional, Tuple
import textwrap
from pinecone import Pinecone
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

class IntegratedPineconeVisualizer:
    def __init__(self, api_key: str, environment: str = None):
        """
        Initialize the integrated Pinecone visualizer
        
        Args:
            api_key: Your Pinecone API key
            environment: Pinecone environment (optional for newer versions)
        """
        self.pc = Pinecone(api_key=api_key)
        self.vectors_df = None
        self.metadata_df = None
        self.combined_df = None
        self.reduced_data = {}
        self.vector_columns = []
        self.metadata_columns = []
        self.vector_matrix = None
        
    def fetch_vectors_with_metadata(
        self, 
        index_name: str,
        metadata_filter: Dict[str, Any] = None,
        top_k: int = 1000,
        include_values: bool = True,
        include_metadata: bool = True,
        namespace: str = "",
        vector_ids: List[str] = None
    ) -> List[Dict]:
        """
        Fetch vectors from Pinecone with metadata filtering
        """
        try:
            # Connect to the index
            index = self.pc.Index(index_name)
            
            all_vectors = []
            
            if vector_ids:
                # Fetch specific vectors by ID
                response = index.fetch(ids=vector_ids, namespace=namespace)
                for vec_id, vec_data in response['vectors'].items():
                    vector_entry = {
                        'id': vec_id,
                        'values': vec_data.get('values', []) if include_values else [],
                        'metadata': vec_data.get('metadata', {}) if include_metadata else {}
                    }
                    all_vectors.append(vector_entry)
            else:
                # Query vectors with metadata filter
                # Get index stats to understand dimensions
                stats = index.describe_index_stats()
                dimension = stats.get('dimension', 3072)  # Default to 3072 for text-embedding-3-large
                
                # Create a dummy query vector (zeros)
                dummy_vector = [0.0] * dimension
                
                # Query with metadata filter
                response = index.query(
                    vector=dummy_vector,
                    filter=metadata_filter,
                    top_k=top_k,
                    include_values=include_values,
                    include_metadata=include_metadata,
                    namespace=namespace
                )
                
                for match in response['matches']:
                    vector_entry = {
                        'id': match['id'],
                        'score': match.get('score', 0.0),
                        'values': match.get('values', []) if include_values else [],
                        'metadata': match.get('metadata', {}) if include_metadata else {}
                    }
                    all_vectors.append(vector_entry)
            
            return all_vectors
            
        except Exception as e:
            print(f"Error fetching vectors: {str(e)}")
            return []
    
    def process_vectors_to_dataframes(
        self, 
        vectors: List[Dict], 
        vector_column_prefix: str = "dim_",
        flatten_metadata: bool = True
    ):
        """
        Convert vector data to separate DataFrames and prepare for visualization
        """
        if not vectors:
            print("No vectors to process")
            return
        
        # Prepare vector data
        vector_data = []
        metadata_data = []
        
        for vector in vectors:
            vector_id = vector['id']
            
            # Process vector values
            vector_row = {'id': vector_id}
            
            if 'score' in vector:
                vector_row['score'] = vector['score']
            
            values = vector.get('values', [])
            if values:
                for i, val in enumerate(values):
                    vector_row[f'{vector_column_prefix}{i}'] = val
            
            vector_data.append(vector_row)
            
            # Process metadata with special handling for group_description
            metadata_row = {'id': vector_id}
            metadata = vector.get('metadata', {})
            
            if metadata:
                if flatten_metadata:
                    self._flatten_metadata(metadata, metadata_row)
                else:
                    metadata_row['metadata_json'] = json.dumps(metadata)
                
                # Ensure group_description is easily accessible
                if 'group_description' in metadata:
                    metadata_row['group_description'] = metadata['group_description']
            
            metadata_data.append(metadata_row)
        
        # Create DataFrames
        self.vectors_df = pd.DataFrame(vector_data)
        self.metadata_df = pd.DataFrame(metadata_data)
        
        # Merge DataFrames
        self.combined_df = pd.merge(self.vectors_df, self.metadata_df, on='id', how='inner')
        
        # Extract column information
        self.vector_columns = [col for col in self.vectors_df.columns if col.startswith(vector_column_prefix)]
        self.metadata_columns = [col for col in self.metadata_df.columns if col != 'id']
        
        # Get vector matrix
        self.vector_matrix = self.vectors_df[self.vector_columns].values
        
        print(f"Processed {len(vectors)} vectors with {len(self.vector_columns)} dimensions")
        print(f"Metadata fields: {self.metadata_columns}")
        
        # Highlight group_id availability  
        if 'group_id' in self.metadata_columns:
            unique_groups = self.combined_df['group_id'].nunique()
            print(f"Found {unique_groups} unique group IDs")
        else:
            print("Note: group_id not found in metadata")
        
        # Highlight claim_text availability
        if 'claim_text' in self.metadata_columns:
            print("Found claim_text for hover display")
        else:
            print("Note: claim_text not found in metadata")
    
    def _flatten_metadata(self, metadata: Dict[str, Any], target_dict: Dict[str, Any], prefix: str = ""):
        """Recursively flatten nested metadata dictionaries"""
        for key, value in metadata.items():
            column_name = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                self._flatten_metadata(value, target_dict, f"{column_name}_")
            elif isinstance(value, list):
                # Convert lists to JSON strings or join simple lists
                if value and all(isinstance(item, (str, int, float)) for item in value):
                    # Simple list - join with semicolons
                    target_dict[column_name] = ';'.join(str(item) for item in value)
                else:
                    # Complex list - convert to JSON
                    target_dict[column_name] = json.dumps(value)
            else:
                # Simple value
                target_dict[column_name] = value

    def _wrap_for_hover(self, value: Any, width: int = 80, max_lines: int = 8) -> str:
        """Insert <br> line breaks into long strings for nicer hover labels."""
        if not isinstance(value, str) or value == "":
            return ""
        lines = textwrap.wrap(value, width=width, break_long_words=False, replace_whitespace=False)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1].rstrip() + " …"
        return "<br>".join(lines)
    
    def export_to_tsvs(
        self, 
        vectors_file: str = "pinecone_vectors.tsv",
        metadata_file: str = "pinecone_metadata.tsv"
    ) -> Tuple[str, str]:
        """Export processed data to TSV files"""
        if self.vectors_df is None or self.metadata_df is None:
            print("No data to export. Please fetch and process vectors first.")
            return None, None
        
        # Export to TSV
        self.vectors_df.to_csv(vectors_file, sep='\t', index=False)
        self.metadata_df.to_csv(metadata_file, sep='\t', index=False)
        
        print(f"Exported vectors to {vectors_file}")
        print(f"Exported metadata to {metadata_file}")
        
        return vectors_file, metadata_file
    
    def reduce_dimensions(self, method='pca', n_components=3, **kwargs):
        """Reduce vector dimensions to 3D using various methods"""
        if self.vector_matrix is None:
            print("No vector data available. Please fetch and process vectors first.")
            return None
        
        print(f"Reducing dimensions using {method.upper()}...")
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_vectors = scaler.fit_transform(self.vector_matrix)
        
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components, **kwargs)
            reduced = reducer.fit_transform(scaled_vectors)
            
            # Calculate explained variance
            explained_var = reducer.explained_variance_ratio_
            print(f"Explained variance: {explained_var}")
            print(f"Total explained variance: {explained_var.sum():.3f}")
            
        elif method.lower() == 'tsne':
            # t-SNE parameters
            default_params = {
                'n_components': n_components,
                'perplexity': min(30, len(self.vector_matrix) - 1),
                'random_state': 42,
                'max_iter': 1000
            }
            default_params.update(kwargs)
            
            reducer = TSNE(**default_params)
            reduced = reducer.fit_transform(scaled_vectors)
            
        elif method.lower() == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not available. Install with: pip install umap-learn")
            
            default_params = {
                'n_components': n_components,
                'random_state': 42,
                'n_neighbors': min(15, len(self.vector_matrix) - 1)
            }
            default_params.update(kwargs)
            
            reducer = umap.UMAP(**default_params)
            reduced = reducer.fit_transform(scaled_vectors)
            
        else:
            raise ValueError("Method must be 'pca', 'tsne', or 'umap'")
        
        self.reduced_data[method] = {
            'data': reduced,
            'reducer': reducer,
            'method': method
        }
        
        return reduced
    
    def build_color_map(self, categories: List[Any]) -> Dict[Any, str]:
        """Create a stable color mapping for categorical values."""
        if categories is None:
            return {}
        unique_vals = list(pd.Series(categories).unique())
        palette = (
            px.colors.qualitative.Alphabet
            + px.colors.qualitative.Safe
            + px.colors.qualitative.Dark24
            + px.colors.qualitative.Pastel
        )
        return {val: palette[i % len(palette)] for i, val in enumerate(unique_vals)}

    def create_3d_plot(self, method='pca', color_by='group_id', size_by=None, 
                       title=None, hover_data=None, save_html=None, show_group_labels=False,
                       show_cluster_regions: bool = False, cluster_opacity: float = 0.10, cluster_alphahull: int = 8,
                       display: bool = True, color_map: Optional[Dict[Any, str]] = None,
                       filter_ids: Optional[List[str]] = None):
        """
        Create interactive 3D scatter plot with emphasis on group_id, showing claim_text on hover
        """
        needs_reduce = (
            method not in self.reduced_data
            or self.reduced_data[method]['data'].shape[0] != len(self.combined_df)
        )
        if needs_reduce:
            self.reduce_dimensions(method)
        
        reduced = self.reduced_data[method]['data']
        
        # Create DataFrame for plotting (start with full set)
        plot_df = self.combined_df.copy()

        # Optional filtering by ids: create a mask then subset both reduced and dataframe
        if filter_ids is not None:
            mask = plot_df['id'].isin(filter_ids)
            reduced = reduced[mask.to_numpy()]
            plot_df = plot_df.loc[mask]

        plot_df['x'] = reduced[:, 0]
        plot_df['y'] = reduced[:, 1]
        plot_df['z'] = reduced[:, 2]
        
        # Set up hover data with claim_text prominently featured (wrapped)
        if hover_data is None:
            hover_data = ['id']
            if 'claim_text' in plot_df.columns:
                plot_df['claim_text_wrapped'] = plot_df['claim_text'].astype(str).apply(lambda s: self._wrap_for_hover(s, width=80, max_lines=8))
                hover_data.append('claim_text_wrapped')
            if 'group_id' in plot_df.columns:
                hover_data.append('group_id')
        
        # Default to coloring by group_id if available
        if color_by == 'group_id' and 'group_id' not in plot_df.columns:
            print("group_id not found, using default coloring")
            color_by = None
        
        # Prepare a stable color map so meshes match point colors
        local_color_map = color_map
        if local_color_map is None and 'group_id' in plot_df.columns:
            local_color_map = self.build_color_map(plot_df['group_id'])

        scatter_kwargs = {}
        if color_by == 'group_id' and local_color_map is not None:
            scatter_kwargs['color_discrete_map'] = local_color_map

        # Create the plot
        if color_by and color_by in plot_df.columns:
            fig = px.scatter_3d(
                plot_df, x='x', y='y', z='z',
                color=color_by,
                size=size_by if size_by and size_by in plot_df.columns else None,
                hover_data=hover_data,
                labels={
                    'x': f'{method.upper()} Component 1',
                    'y': f'{method.upper()} Component 2',
                    'z': f'{method.upper()} Component 3'
                },
                **scatter_kwargs
            )
            
            # Customize colors for group_id
            if color_by == 'group_id':
                fig.update_traces(
                    marker=dict(
                        size=8,
                        line=dict(width=1, color='white'),
                        opacity=0.8
                    )
                )
        else:
            fig = px.scatter_3d(
                plot_df, x='x', y='y', z='z',
                hover_data=hover_data,
                labels={
                    'x': f'{method.upper()} Component 1',
                    'y': f'{method.upper()} Component 2',
                    'z': f'{method.upper()} Component 3'
                },
                **scatter_kwargs
            )
        
        # Add group labels if requested and group_id is available
        if show_group_labels and 'group_id' in plot_df.columns:
            # Calculate centroids for each group
            group_centroids = plot_df.groupby('group_id')[['x', 'y', 'z']].mean()
            
            # Add text annotations for group centroids
            for group_name, centroid in group_centroids.iterrows():
                fig.add_trace(go.Scatter3d(
                    x=[centroid['x']],
                    y=[centroid['y']],
                    z=[centroid['z']],
                    mode='markers',
                    marker=dict(size=3, color='black'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Optionally add faded cluster regions per group using Mesh3d (initially hidden unless requested)
        mesh_indices = []
        if 'group_id' in plot_df.columns and len(plot_df) >= 4:
            for gid, g in plot_df.groupby('group_id'):
                if len(g) < 4:
                    continue
                mesh_color = local_color_map.get(gid, '#95a5a6') if local_color_map else '#95a5a6'
                fig.add_trace(go.Mesh3d(
                    x=g['x'], y=g['y'], z=g['z'],
                    alphahull=cluster_alphahull,
                    color=mesh_color,
                    opacity=cluster_opacity,
                    hoverinfo='skip',
                    visible=bool(show_cluster_regions),
                    name=f'Region {gid}'
                ))
                mesh_indices.append(len(fig.data) - 1)

        # Build summary annotation text (top, no legend/title)
        num_points = len(plot_df)
        num_groups = plot_df['group_id'].nunique() if 'group_id' in plot_df.columns else None
        summary_bits = [f"{num_points} points"]
        if num_groups is not None:
            summary_bits.append(f"{num_groups} groups")
        reducer = self.reduced_data[method].get('reducer')
        if method.lower() == 'pca' and hasattr(reducer, 'explained_variance_ratio_'):
            try:
                summary_bits.append(f"PCA {sum(reducer.explained_variance_ratio_):.1%} var")
            except Exception:
                pass
        summary_text = " · ".join(summary_bits)

        # Customize layout
        fig.update_layout(
            scene=dict(
                xaxis_title=f'{method.upper()} Component 1',
                yaxis_title=f'{method.upper()} Component 2',
                zaxis_title=f'{method.upper()} Component 3',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                # Keep axes fixed (no moving walls) and apply static scene background
                bgcolor="rgba(245, 247, 250, 1)",
                xaxis=dict(
                    showbackground=False,
                    gridcolor="#e9edf2",
                    showspikes=False,
                    spikesides=False
                ),
                yaxis=dict(
                    showbackground=False,
                    gridcolor="#e9edf2",
                    showspikes=False,
                    spikesides=False
                ),
                zaxis=dict(
                    showbackground=False,
                    gridcolor="#e9edf2",
                    showspikes=False,
                    spikesides=False
                )
            ),
            width=1920,
            height=1080,
            dragmode='orbit',  # Only rotate objects, not axes
            showlegend=False,
            paper_bgcolor='rgba(245, 247, 250, 1)'
        )

        # Add top summary annotation
        fig.add_annotation(
            x=0.5, y=1.06, xref='paper', yref='paper',
            text=summary_text,
            showarrow=False,
            font=dict(size=16, color='#2c3e50'),
            align='center'
        )

        # Toggle buttons for cluster regions
        if mesh_indices:
            fig.update_layout(
                updatemenus=[dict(
                    type='buttons',
                    direction='right',
                    x=0.01, y=1.06, xanchor='left', yanchor='top',
                    buttons=[
                        dict(label='Regions Off', method='restyle', args=[{'visible': False}, mesh_indices]),
                        dict(label='Regions On', method='restyle', args=[{'visible': True}, mesh_indices])
                    ]
                )]
            )
        
        if save_html:
            fig.write_html(save_html)
            print(f"Interactive plot saved to {save_html}")
        
        if display:
            fig.show()
        return fig
    
    def analyze_group_distributions(self, method='pca'):
        """Analyze how group_id values are distributed in 3D space"""
        if 'group_id' not in self.combined_df.columns:
            print("group_id not found in metadata")
            return
        
        if method not in self.reduced_data:
            self.reduce_dimensions(method)
        
        reduced = self.reduced_data[method]['data']
        
        # Create analysis dataframe
        analysis_df = self.combined_df.copy()
        analysis_df['x'] = reduced[:, 0]
        analysis_df['y'] = reduced[:, 1]
        analysis_df['z'] = reduced[:, 2]
        
        print("=== Group ID Analysis ===")
        
        # Group statistics
        group_stats = analysis_df.groupby('group_id').agg({
            'x': ['mean', 'std', 'count'],
            'y': ['mean', 'std'],
            'z': ['mean', 'std']
        }).round(3)
        
        print("\nGroup Statistics:")
        print(group_stats)
        
        # Calculate distances between group centroids
        centroids = analysis_df.groupby('group_id')[['x', 'y', 'z']].mean()
        
        print(f"\nFound {len(centroids)} unique groups:")
        for i, (group_name, centroid) in enumerate(centroids.iterrows()):
            count = analysis_df[analysis_df['group_id'] == group_name].shape[0]
            print(f"{i+1}. Group {group_name}: {count} vectors")
        
        # Create group-focused visualization
        hover_data_analysis = ['id', 'group_id']
        if 'claim_text' in analysis_df.columns:
            analysis_df['claim_text_wrapped'] = analysis_df['claim_text'].astype(str).apply(lambda s: self._wrap_for_hover(s, width=80, max_lines=8))
            hover_data_analysis.append('claim_text_wrapped')
        
        fig = px.scatter_3d(
            analysis_df, x='x', y='y', z='z',
            color='group_id',
            hover_data=hover_data_analysis,
            size_max=10
        )
        
        # Do not render centroid markers to keep the plot clean
        
        # Build and add summary annotation
        num_points = len(analysis_df)
        num_groups = analysis_df['group_id'].nunique()
        summary_text = f"{num_points} points · {num_groups} groups · {method.upper()}"

        fig.update_layout(
            width=1920,
            height=1080,
            dragmode='orbit',  # Only rotate objects, not axes
            showlegend=False,
            paper_bgcolor='rgba(245, 247, 250, 1)',
            scene=dict(
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                bgcolor='rgba(245, 247, 250, 1)',
                xaxis=dict(showbackground=False, gridcolor='#e9edf2', showspikes=False, spikesides=False),
                yaxis=dict(showbackground=False, gridcolor='#e9edf2', showspikes=False, spikesides=False),
                zaxis=dict(showbackground=False, gridcolor='#e9edf2', showspikes=False, spikesides=False)
            )
        )
        fig.add_annotation(x=0.5, y=1.06, xref='paper', yref='paper', text=summary_text,
                           showarrow=False, font=dict(size=16, color='#2c3e50'), align='center')
        
        fig.show()
        return analysis_df, centroids
    
    def fetch_and_visualize(
        self,
        index_name: str,
        metadata_filter: Dict[str, Any] = None,
        top_k: int = 1000,
        methods: List[str] = ['pca', 'tsne'],
        export_tsvs: bool = True,
        save_html: bool = True,
        use_density_clustering: bool = True,
        density_eps: float = None,
        clustering_comparison: bool = True
    ):
        """
        Complete workflow: fetch from Pinecone -> process -> visualize
        
        Args:
            use_density_clustering: Whether to perform density-based clustering analysis
            density_eps: Specific eps value for density clustering (None for auto-tune)
            clustering_comparison: Whether to show clustering method comparisons
        """
        print("=== Starting Integrated Pinecone Visualization Workflow ===")
        print("Optimized for text-embedding-3-large (3072 dimensions)")
        
        # Step 1: Fetch vectors
        print(f"\n1. Fetching vectors from index '{index_name}'...")
        vectors = self.fetch_vectors_with_metadata(
            index_name=index_name,
            metadata_filter=metadata_filter,
            top_k=top_k,
            include_values=True,
            include_metadata=True
        )
        
        if not vectors:
            print("No vectors found. Check your index name and filter.")
            return
        
        # Step 2: Process data
        print(f"\n2. Processing {len(vectors)} vectors...")
        self.process_vectors_to_dataframes(vectors)
        
        # Step 3: Export TSVs (optional)
        if export_tsvs:
            print(f"\n3. Exporting to TSV files...")
            self.export_to_tsvs()
        
        # Step 4: Create basic visualizations
        print(f"\n4. Creating 3D visualizations...")
        
        for method in methods:
            print(f"\nCreating {method.upper()} visualization...")
            
            html_filename = f"vectors_{method}_3d.html" if save_html else None
            
            self.create_3d_plot(
                method=method,
                color_by='group_id',
                title=f'Vector Groups - {method.upper()}',
                save_html=html_filename,
                show_group_labels=False,
                display=True
            )
        
        # Step 5: Analyze group distributions
        print(f"\n5. Analyzing group distributions...")
        self.analyze_group_distributions(method=methods[0])
        
        # Step 6: Clustering Analysis
        if clustering_comparison:
            print(f"\n6. Performing clustering analysis...")
            
            # K-means clustering
            print("\n--- K-Means Clustering ---")
            kmeans_clusters, kmeans_df = self.analyze_clusters(method=methods[0])
            
            # Density-based clustering
            if use_density_clustering:
                print("\n--- Density-Based Clustering ---")
                density_clusters, density_df, cluster_stats = self.analyze_density_clusters(
                    method=methods[0], 
                    eps=density_eps,
                    auto_tune=True
                )
                
                # Create density comparison with different thresholds
                print("\n--- Density Threshold Comparison ---")
                self.create_density_comparison(method=methods[0])
        
        print(f"\n=== Workflow Complete ===")
        print(f"✓ Processed {len(vectors)} vectors with 3072-dimensional embeddings")
        print(f"✓ Created {len(methods)} basic visualizations")
        if clustering_comparison:
            print("✓ Performed K-means and density-based clustering analysis")
            if use_density_clustering:
                print("✓ Generated density threshold comparisons")
        if export_tsvs:
            print("✓ Exported TSV files")
        if save_html:
            print("✓ Saved interactive HTML files")
        
        # Summary insights
        if hasattr(self, 'combined_df') and 'group_id' in self.combined_df.columns:
            n_groups = self.combined_df['group_id'].nunique()
            print(f"✓ Found {n_groups} unique group IDs in the data")
    
    def interactive_density_tuning(self, method='pca'):
        """
        Interactive function to tune density parameters and see results
        """
        if method not in self.reduced_data:
            self.reduce_dimensions(method)
        
        print("=== Interactive Density Tuning ===")
        print("Try different eps values to find meaningful clusters")
        print("Lower eps = tighter clusters, higher eps = looser clusters")
        
        # Get recommended starting point
        recommended_eps, _ = self.find_optimal_density_params(method)
        print(f"Recommended starting eps: {recommended_eps:.3f}")
        
        # Suggest testing range
        test_values = [
            recommended_eps * 0.3,
            recommended_eps * 0.6,
            recommended_eps,
            recommended_eps * 1.5,
            recommended_eps * 2.0
        ]
        
        print(f"\nSuggested eps values to try: {[f'{v:.3f}' for v in test_values]}")
        print("\nCall analyze_density_clusters(eps=YOUR_VALUE) to test specific values")
        print("Call create_density_comparison(eps_values=[val1, val2, val3, val4]) for side-by-side comparison")
        
        return recommended_eps, test_values

def main():
    """
    Example usage of the IntegratedPineconeVisualizer with density clustering
    """
    # Load configuration from environment variables
    API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "content-gen-claim-index")
    BRAND_ID = os.getenv("BRAND_ID")
    
    if not API_KEY:
        print("❌ Error: PINECONE_API_KEY not found")
        print("Please create a .env file with your Pinecone API key:")
        print("  PINECONE_API_KEY=your-api-key-here")
        print("  PINECONE_INDEX_NAME=your-index-name")
        print("  BRAND_ID=your-brand-id (optional)")
        return
    
    # Initialize the integrated visualizer
    visualizer = IntegratedPineconeVisualizer(api_key=API_KEY)
    
    # Example metadata filters for text-embedding-3-large vectors
    metadata_filters = {}
    if BRAND_ID:
        metadata_filters = {
            "brand_id": {"$eq": BRAND_ID}
        }
    
    # You can customize filters as needed:
    # metadata_filters = {
    #     # Example: Filter by specific group
    #     "group_id": "specific_group_id"
    #     
    #     # Example: Multiple conditions
    #     "$and": [
    #         {"group_id": {"$exists": True}},
    #         {"category": "documents"},
    #         {"created_date": {"$gte": "2024-01-01"}}
    #     ]
    # }
    
    print("=== Text-Embedding-3-Large Vector Analysis ===")
    print("This program is optimized for 3072-dimensional embeddings")
    
    # Run complete workflow with density clustering
    visualizer.fetch_and_visualize(
        index_name=INDEX_NAME,
        metadata_filter=metadata_filters,
        top_k=1000,  # Adjust based on your needs
        methods=['pca', 'tsne'],  # Add 'umap' if installed
        export_tsvs=True,
        save_html=True,
        use_density_clustering=True,  # Enable density clustering
        density_eps=None,  # Auto-tune the density threshold
        clustering_comparison=True  # Show clustering comparisons
    )
    
    # Optional: Interactive density tuning
    print("\n=== Interactive Density Tuning ===")
    print("For fine-tuning density parameters, you can use:")
    print("recommended_eps, test_values = visualizer.interactive_density_tuning()")
    print("visualizer.analyze_density_clusters(eps=YOUR_CUSTOM_VALUE)")

# Example of advanced density analysis
def advanced_density_analysis():
    """
    Example of more advanced density clustering analysis
    """
    API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = "your-index-name"
    
    if not API_KEY:
        print("Please set PINECONE_API_KEY environment variable")
        return
    
    visualizer = IntegratedPineconeVisualizer(api_key=API_KEY)
    
    # Fetch a smaller dataset for detailed analysis
    vectors = visualizer.fetch_vectors_with_metadata(
        index_name=INDEX_NAME,
        metadata_filter={"group_description": {"$exists": True}},
        top_k=500  # Smaller dataset for detailed analysis
    )
    
    visualizer.process_vectors_to_dataframes(vectors)
    
    # Try different dimensionality reduction methods with density clustering
    methods = ['pca', 'tsne']
    
    for method in methods:
        print(f"\n=== {method.upper()} Density Analysis ===")
        
        # Get interactive tuning suggestions
        recommended_eps, test_values = visualizer.interactive_density_tuning(method)
        
        # Test multiple density thresholds
        print(f"Testing density thresholds for {method.upper()}...")
        
        for eps in test_values[:3]:  # Test first 3 suggested values
            print(f"\n--- Testing eps={eps:.3f} ---")
            clusters, df, stats = visualizer.analyze_density_clusters(
                method=method, 
                eps=eps, 
                auto_tune=False
            )
        
        # Create comparison visualization
        visualizer.create_density_comparison(method, test_values)

if __name__ == "__main__":
    main()
    
    # Uncomment to run advanced analysis
    # advanced_density_analysis()