# Quick Start Guide - Dataset Search & Selection

## Installation

1. Install dependencies:
```bash
pip install duckdb pandas sentence-transformers requests PyYAML datasets pyarrow huggingface_hub keyring cryptography python-dotenv
```

2. Place the files in your project directory:
```
your-project/
├── ddb_dataset_sel.py        # New dataset configurator
├── ddbrag_updated.py          # Updated main application
├── ddb_sp_mdl_sel.py          # Existing model configurator
├── hf_auth.py                 # Updated authentication
├── ddrag_model.py             # Fixed model manager
├── select_proc.py             # Existing processor detection
└── requirements.txt           # Dependencies
```

## Basic Usage

### Option 1: Quick Start (2 steps)
```bash
# 1. Run the application
python ddbrag_updated.py

# 2. Use the quick setup options
Model Setup: 2  # Quick setup
Dataset Setup: 2  # Legacy method
```

### Option 2: Full Featured (with dataset search)
```bash
# 1. Run the application
python ddbrag_updated.py

# 2. Choose advanced configurators
Model Setup: 1
Dataset Setup: 1

# 3. Search for model
Search: e5

# 4. Search for dataset
Search: qa

# 5. Configure and start querying
```

## Standalone Dataset Search

Search and load a dataset without running the full RAG system:

```bash
python ddb_dataset_sel.py
```

## Common Workflows

### Workflow 1: Load Pre-configured Dataset
```python
from ddb_dataset_sel import DatasetConfigurator

# Initialize
configurator = DatasetConfigurator()

# Load existing config
config = configurator.load_config_file()

# Load dataset with saved config
dataset = configurator.load_dataset(
    config['dataset_name'], 
    config['parameters']
)
```

### Workflow 2: Search and Load New Dataset
```python
from ddb_dataset_sel import DatasetConfigurator

# Initialize and run full workflow
configurator = DatasetConfigurator()
dataset, config = configurator.run()

# Dataset is now loaded and cached
print(f"Loaded: {config['dataset_name']}")
print(f"Rows: {len(dataset)}")
```

### Workflow 3: Multi-Dataset Management
```bash
# Start application
python ddbrag_updated.py

# Load first dataset
Dataset Setup: 1
Search: legal thai
# Configure and load

# In interactive mode, switch datasets
Query: /switch
# Select different dataset

Query: /list
# View all loaded datasets
```

## Interactive Commands Reference

Once in interactive query mode:

| Command | Description |
|---------|-------------|
| `<text>` | Run similarity search with your query |
| `/switch` | Switch to a different loaded dataset |
| `/list` | List all loaded datasets with details |
| `/embed` | Create embeddings for current dataset |
| `/load` | Load a new dataset (restart required) |
| `/quit` | Exit interactive mode |

## Configuration Files

### Dataset Config Example
```yaml
# ~/.config/ddbrag/configs/dataset_mydata-102324-1430.yaml
dataset_name: username/my-dataset
created: '2024-10-23T14:30:00'
parameters:
  cache_dir: /home/user/.config/ddbrag/datasets/cache
  split: train
  sample_size: 10000
  save_format: parquet
  use_duckdb: true
```

### Loading Saved Config
```bash
python ddbrag_updated.py
Dataset Setup: 1
Options: L  # Load existing config
Path: dataset_mydata-102324-1430.yaml
```

## Troubleshooting

### Issue: "No datasets found"
**Solution**: Try broader search terms or check your network connection

### Issue: "Authentication required"
**Solution**: 
```bash
# Login to HuggingFace
python hf_auth.py --login

# Or during application startup, choose to login when prompted
```

### Issue: "Could not find parquet files"
**Solution**: Some datasets use different structures
1. Try disabling `use_duckdb` in parameters
2. The system will fall back to the datasets library

### Issue: "Out of memory"
**Solution**: Use the `sample_size` parameter to load a subset:
```
Parameters to edit: 3
New value: 5000
```

## Advanced Features

### Custom Dataset Paths
Load datasets from custom HuggingFace paths:
```python
# For datasets in subdirectories
dataset_name = "organization/dataset-name"
# System will try multiple common patterns automatically
```

### Batch Processing
Create embeddings for large datasets efficiently:
```python
# The system automatically:
# - Shows progress every 100 rows
# - Handles encoding errors gracefully
# - Removes failed encodings
```

### Multiple Format Support
Cache in your preferred format:
- **Parquet**: Fast, compressed (recommended)
- **CSV**: Universal compatibility
- **Pickle**: Python-native, preserves types

## Tips for Best Performance

1. **Use DuckDB loading** for large parquet datasets (enabled by default)
2. **Enable caching** to avoid re-downloading (enabled by default)
3. **Sample first** with small sample_size when testing new datasets
4. **Create embeddings once** and save them for reuse
5. **Use parquet format** for cache (fastest and most efficient)

## Next Steps

1. ✅ Install dependencies
2. ✅ Run the application
3. ✅ Search and load a dataset
4. ✅ Create embeddings
5. ✅ Start querying
6. 📚 Read full documentation in README_DATASET_FEATURE.md
7. 🔧 Explore advanced features
8. 🚀 Build your RAG application

## Support

For detailed documentation, see:
- `README_DATASET_FEATURE.md` - Complete feature documentation
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details

## Example Session

```
$ python ddbrag_updated.py

================================================================================
                          DuckDB RAG System
================================================================================

Model Setup Options:
1. Advanced Model Configurator (search, configure, save settings)
2. Quick Setup (use predefined models)

Choice (1-2, default=1): 1

>>> Model Search

Enter a search term to find models (e.g., 'gemma', 'qwen', 'e5')
Or type 'exit' to quit

Search: e5

>>> Available Models

1. intfloat/multilingual-e5-large
   Downloads: 1,234,567 | Likes: 456
   Multilingual embedding model optimized for retrieval tasks

2. intfloat/e5-large-v2
   Downloads: 987,654 | Likes: 321
   English embedding model with strong performance

Your choice: 1
✓ Model loaded successfully!

Dataset Setup Options:
1. Advanced Dataset Configurator (search HuggingFace datasets)
2. Legacy method (use default Thai legal dataset or local pickle)

Choice (1-2, default=1): 1

>>> Dataset Search

Enter a search term to find datasets (e.g., 'legal', 'qa', 'thai')
Or type 'exit' to quit

Search: qa

✓ Found 15 matching datasets

[Select dataset, configure parameters, load...]

✓ Dataset loaded successfully!
✓ Embeddings created for 5000 rows

>>> Interactive Query Mode

Commands:
  <query>  - Search the dataset
  /switch  - Switch to a different dataset
  /list    - List all loaded datasets
  /embed   - Create embeddings for current dataset
  /load    - Load a new dataset
  /quit    - Exit interactive mode

Current dataset: username/qa-dataset

Query: machine learning basics
✓ Query encoded
✓ Found 5 results

================================================================================
                              SEARCH RESULTS
================================================================================

Rank 1 (Distance: 0.1234)

Question:
  What are the fundamentals of machine learning?

Answer:
  Machine learning is a subset of artificial intelligence that focuses on...

[More results...]

Query: /quit
✓ Session complete!
```

Happy RAG building! 🚀
