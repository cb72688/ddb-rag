# DuckDB RAG System - Dataset Search & Selection Feature

## Overview
This update adds comprehensive dataset search, selection, and management capabilities to the DuckDB RAG system, mirroring the existing model configurator functionality.

## New Module: `ddb_dataset_sel.py`

### Features
- **Search HuggingFace Datasets**: Search and browse datasets from the HuggingFace Hub
- **Interactive Selection**: Paginated display of search results with detailed information
- **Configuration Management**: Save and load dataset configurations
- **Multiple Datasets**: Support for loading and managing multiple datasets simultaneously
- **Caching**: Automatic caching of downloaded datasets in multiple formats (parquet, csv, pickle)
- **Authentication**: Integrated authentication for gated datasets
- **Flexible Loading**: Uses DuckDB for fast loading or falls back to the datasets library

### Usage

#### Standalone Usage
```bash
python ddb_dataset_sel.py
```

#### Integrated Usage
The dataset configurator is integrated into the main RAG system (`ddbrag_updated.py`):

1. Run the main application:
   ```bash
   python ddbrag_updated.py
   ```

2. Choose dataset setup option when prompted:
   - Option 1: Advanced Dataset Configurator (search HuggingFace)
   - Option 2: Legacy method (default Thai legal dataset)

3. Search for datasets by keywords (e.g., "legal", "qa", "thai")

4. Select a dataset from search results

5. Configure loading parameters:
   - Cache directory
   - Split (train/test/validation)
   - Sample size
   - Save format (parquet/csv/pickle)
   - DuckDB usage

6. Dataset is automatically downloaded, cached, and loaded

### Key Classes

#### DatasetConfigurator
Main class for dataset search and configuration:
- `search_huggingface_datasets(query)`: Search HF for datasets
- `display_datasets_paginated(datasets)`: Interactive selection UI
- `configure_parameters(dataset_name)`: Configure loading parameters
- `load_dataset(dataset_name, params)`: Load and cache dataset
- `run()`: Main workflow orchestration

## Updated Main Application: `ddbrag_updated.py`

### New Features
- **Multi-Dataset Support**: Load and manage multiple datasets simultaneously
- **Dataset Switching**: Switch between loaded datasets during interactive queries
- **On-Demand Embeddings**: Create embeddings for any text column in any dataset
- **Enhanced Interactive Mode**: New commands for dataset management

### Interactive Commands
- `<query>` - Search the current dataset
- `/switch` - Switch to a different loaded dataset
- `/list` - List all loaded datasets
- `/embed` - Create embeddings for current dataset
- `/load` - Load a new dataset
- `/quit` - Exit interactive mode

### Example Workflow

```python
# 1. Start the application
python ddbrag_updated.py

# 2. Choose Advanced Model Configurator
Choice: 1

# Search for and select an embedding model
Search: e5
# Select from results...

# 3. Choose Advanced Dataset Configurator
Choice: 1

# Search for datasets
Search: legal thai
# Select from results...

# 4. Configure dataset loading
# Choose split, sample size, cache settings...

# 5. Create embeddings if needed
Create embeddings now? y
Select column to embed: question

# 6. Run queries
Query: สัญญาจ้างงาน
# Results displayed...

# 7. Switch datasets if multiple loaded
Query: /switch
# Select different dataset...

# 8. Query new dataset
Query: labor laws
# Results from new dataset...
```

## Configuration Files

### Dataset Configuration Format (YAML)
```yaml
dataset_name: username/dataset-name
created: '2024-01-01T12:00:00'
parameters:
  cache_dir: /path/to/cache
  split: train
  sample_size: null
  save_format: parquet
  use_duckdb: true
```

### Default Locations
- Configurations: `~/.config/ddbrag/configs/`
- Dataset cache: `~/.config/ddbrag/datasets/cache/`
- Logs: `./logs/`

## Key Improvements

### Code Reuse
- Reuses `Colors` class for consistent formatting
- Reuses authentication patterns from `hf_auth` module
- Mirrors parameter configuration workflow from model configurator
- Maintains consistent user experience across features

### Performance Optimizations
- DuckDB for fast parquet loading
- Automatic caching to avoid re-downloads
- Sample size limits for testing
- Multiple file format support

### Error Handling
- Graceful fallback from DuckDB to datasets library
- Authentication error detection and recovery
- Clear error messages and suggestions
- Path detection for different HF dataset structures

## Dependencies

Required packages (add to requirements.txt):
```
duckdb>=0.9.0
pandas>=2.0.0
datasets>=2.14.0
pyarrow>=14.0.0  # For parquet support
```

## Files Modified/Created

### New Files
- `ddb_dataset_sel.py` - Dataset configurator module
- `README_DATASET_FEATURE.md` - This documentation

### Updated Files
- `ddbrag_updated.py` - Main application with dataset integration
- `hf_auth.py` - Fixed syntax errors (0x600 -> 0o600)
- `ddrag_model.py` - Fixed syntax errors (parentheses, typos)

### Original Files (Reference)
- `ddb_sp_mdl_sel.py` - Model configurator (unchanged)
- `ddbrag.py` - Original main application (preserved)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  ddbrag_updated.py                  │
│                 (Main Application)                  │
└────────────┬───────────────────────┬────────────────┘
             │                       │
             ▼                       ▼
┌────────────────────────┐  ┌──────────────────────┐
│  ddb_sp_mdl_sel.py     │  │  ddb_dataset_sel.py  │
│  (Model Configurator)  │  │ (Dataset Config)     │
└────────────┬───────────┘  └──────────┬───────────┘
             │                         │
             └────────┬────────────────┘
                      ▼
             ┌────────────────┐
             │   hf_auth.py   │
             │(Authentication)│
             └────────────────┘
```

## Future Enhancements

Potential additions:
1. Dataset filtering and preprocessing
2. Automatic column type detection for embeddings
3. Multi-column embedding support
4. Dataset merging and joining
5. Export functionality for processed datasets
6. Dataset statistics and analysis tools
7. Version control for dataset configurations
8. Batch dataset loading
9. Dataset recommendation based on query patterns
10. Integration with more data sources (CSV, JSON, SQL databases)

## Troubleshooting

### Common Issues

**Issue**: "Could not find parquet files"
- Solution: Dataset may use different structure. Try disabling `use_duckdb` option.

**Issue**: "Authentication required"
- Solution: Run `python hf_auth.py --login` or use the login prompt in the configurator.

**Issue**: "Permission denied" when caching
- Solution: Check write permissions for `~/.config/ddbrag/datasets/cache/`

**Issue**: "Out of memory" when loading large datasets
- Solution: Use the `sample_size` parameter to load a subset of the data.

## License
Same as the main project.

## Contributors
- Original model configurator pattern by [original author]
- Dataset configurator implementation by [your name]
