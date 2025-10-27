# Embeddings Visualizer 🎨

An interactive 3D visualization tool for exploring high-dimensional vector embeddings stored in Pinecone. Optimized for OpenAI's `text-embedding-3-large` (3072 dimensions) but works with any embedding model.

## Features ✨

- **3D Visualization**: Interactive 3D plots using PCA, t-SNE, and UMAP dimensionality reduction
- **Multiple Interfaces**: 
  - Command-line Python API
  - Streamlit web app with dark mode
  - Dash web app with live controls
- **Group Analysis**: Automatic clustering and group-based coloring
- **Metadata Support**: Rich hover information with claim text and custom metadata
- **Density Clustering**: DBSCAN-based density analysis with auto-tuning
- **Export Options**: Save visualizations as interactive HTML files
- **TSV Export**: Export vectors and metadata to TSV files for further analysis

## Screenshots

### 3D Scatter Plot with Group Coloring
Interactive visualization showing embeddings colored by group_id with customizable camera angles and cluster regions.

### Streamlit Interface
Modern web interface with real-time filtering, search, and dark mode support.

## Installation

### Prerequisites
- Python 3.8 or higher
- A Pinecone account with an existing index
- Pinecone API key

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/embeddings_viz.git
   cd embeddings_viz
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Copy the example environment file and add your credentials:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Pinecone credentials:
   ```env
   PINECONE_API_KEY=your-pinecone-api-key-here
   PINECONE_INDEX_NAME=your-index-name-here
   BRAND_ID=your-brand-id-here  # Optional: for filtering
   ```

## Usage

### 1. Command-Line Interface

Run the main visualization script:

```bash
python3 embedding_visualizer.py
```

This will:
- Fetch vectors from your Pinecone index
- Apply dimensionality reduction (PCA and t-SNE)
- Generate interactive 3D visualizations
- Export data to TSV files
- Perform clustering analysis

### 2. Streamlit Web App

Launch the interactive Streamlit interface:

```bash
streamlit run streamlit_app.py
```

Features:
- Real-time 3D visualization
- Group filtering and search
- Camera presets (Iso, Front, Top)
- Dark mode toggle
- Adjustable point size and opacity
- Export to HTML

Open your browser to `http://localhost:8501`

### 3. Dash Web App

Launch the Dash dashboard:

```bash
python3 dash_app.py
```

Features:
- Similar to Streamlit but with Dash framework
- Multiple camera angles
- Live controls for visualization parameters
- Export functionality

Open your browser to `http://localhost:8090`

## Python API

### Basic Usage

```python
from embedding_visualizer import IntegratedPineconeVisualizer
import os

# Initialize visualizer
api_key = os.getenv("PINECONE_API_KEY")
visualizer = IntegratedPineconeVisualizer(api_key=api_key)

# Fetch vectors with metadata filtering
vectors = visualizer.fetch_vectors_with_metadata(
    index_name="your-index-name",
    metadata_filter={"brand_id": {"$eq": "your-brand-id"}},
    top_k=1000
)

# Process and create visualizations
visualizer.process_vectors_to_dataframes(vectors)
visualizer.create_3d_plot(
    method='pca',
    color_by='group_id',
    save_html='output.html'
)
```

### Advanced Usage

```python
# Analyze group distributions
visualizer.analyze_group_distributions(method='pca')

# Density-based clustering
clusters, df, stats = visualizer.analyze_density_clusters(
    method='pca',
    eps=0.5,
    auto_tune=True
)

# Export to TSV
visualizer.export_to_tsvs(
    vectors_file="vectors.tsv",
    metadata_file="metadata.tsv"
)
```

## Project Structure

```
embeddings_viz/
├── embedding_visualizer.py   # Main visualization library
├── streamlit_app.py          # Streamlit web interface
├── dash_app.py              # Dash web interface
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (not in repo)
├── .env.example            # Example environment file
├── .gitignore             # Git ignore rules
├── README.md             # This file
├── pinecone_vectors.tsv   # Generated: Vector data
├── pinecone_metadata.tsv  # Generated: Metadata
└── vectors_*.html        # Generated: Interactive plots
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `PINECONE_API_KEY` | Your Pinecone API key | Yes |
| `PINECONE_INDEX_NAME` | Name of your Pinecone index | Yes |
| `BRAND_ID` | Brand ID for filtering (optional) | No |

### Metadata Filtering

You can filter vectors using Pinecone's metadata filtering syntax:

```python
# Filter by single field
metadata_filter = {"group_id": {"$eq": "specific_group"}}

# Multiple conditions
metadata_filter = {
    "$and": [
        {"group_id": {"$exists": True}},
        {"category": "documents"},
        {"created_date": {"$gte": "2024-01-01"}}
    ]
}
```

## Dimensionality Reduction Methods

### PCA (Principal Component Analysis)
- Fast and deterministic
- Best for initial exploration
- Shows explained variance

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
- Good for revealing clusters
- Non-linear dimensionality reduction
- Can be slower for large datasets

### UMAP (Uniform Manifold Approximation and Projection)
- Balances speed and quality
- Preserves both local and global structure
- Requires `umap-learn` package

## Expected Metadata Fields

The visualizer works best with the following metadata fields:

- `group_id`: Identifier for grouping related vectors
- `group_description`: Human-readable description of the group
- `claim_text`: Text content displayed on hover
- `brand_id`: For filtering by brand

Additional metadata fields will be automatically flattened and included.

## Output Files

### Generated Visualizations
- `vectors_pca_3d.html`: Interactive PCA visualization
- `vectors_tsne_3d.html`: Interactive t-SNE visualization
- `vectors_umap_3d.html`: Interactive UMAP visualization (if UMAP is installed)

### Exported Data
- `pinecone_vectors.tsv`: Vector embeddings in TSV format
- `pinecone_metadata.tsv`: Metadata in TSV format

## Troubleshooting

### "No vectors found"
- Check your index name is correct
- Verify your API key is valid
- Ensure your metadata filter matches existing data

### UMAP not available
```bash
pip install umap-learn
```

### Performance issues
- Reduce `top_k` parameter for fewer vectors
- Use PCA instead of t-SNE for faster results
- Consider filtering by metadata to reduce dataset size

## Dependencies

Core libraries:
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `plotly`: Interactive visualizations
- `scikit-learn`: Dimensionality reduction and clustering
- `pinecone`: Pinecone client SDK
- `streamlit`: Web interface
- `python-dotenv`: Environment variable management

Optional:
- `umap-learn`: UMAP dimensionality reduction

See `requirements.txt` for complete list with versions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built for OpenAI's `text-embedding-3-large` embeddings
- Powered by Pinecone vector database
- Visualizations created with Plotly

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Note**: Remember to keep your `.env` file secure and never commit it to version control. The `.gitignore` file is configured to exclude it automatically.

