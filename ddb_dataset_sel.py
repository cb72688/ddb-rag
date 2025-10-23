#!/usr/bin/env python3
"""
Dataset Configurator for DuckDB RAG System
Handles dataset search, selection, and loading from Hugging Face
"""

import os
import sys
import json
import yaml
import duckdb
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import requests

# Import authentication module
try:
    from hf_auth import hfAuth
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    print("WARNING: hf_auth module not found. Gated datasets may not be accessible")

# Configuration paths
USER_HOME = os.path.expanduser("~")
CONFIG_BASE = os.path.join(USER_HOME, ".config", "ddbrag")
CONFIG_DIR = os.path.join(CONFIG_BASE, "configs")
DATASET_DIR = os.path.join(CONFIG_BASE, "datasets")
DATASET_CACHE_DIR = os.path.join(DATASET_DIR, "cache")
LOG_DIR = "./logs/"
folders = [CONFIG_DIR, DATASET_DIR, DATASET_CACHE_DIR, LOG_DIR]

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class DatasetConfigurator:
    """Main class for dataset selection and configuration"""
    
    def __init__(self):
        self.selected_dataset = None
        self.dataset_config = {}
        self.config_file_path = None
        self.loaded_datasets = {}  # Cache of loaded datasets
        self.ensure_directories()

        # Initialize authentication
        if AUTH_AVAILABLE:
            self.auth = hfAuth()
        else:
            self.auth = None

    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
        
    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")
        
    def print_section(self, text: str):
        """Print formatted section header"""
        print(f"\n{Colors.OKBLUE}{Colors.BOLD}>>> {text}{Colors.ENDC}")
        
    def print_success(self, text: str):
        """Print success message"""
        print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")
        
    def print_warning(self, text: str):
        """Print warning message"""
        print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")
        
    def print_error(self, text: str):
        """Print error message"""
        print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")
        
    def list_existing_configs(self):
        """List all existing dataset configuration files"""
        self.print_section("Existing Dataset Configurations")
        
        config_files = list(Path(CONFIG_DIR).glob("dataset_*.yaml")) + \
                      list(Path(CONFIG_DIR).glob("dataset_*.json"))
        
        if not config_files:
            print("  No existing dataset configuration files found.")
            return []
        
        print(f"\n  Found {len(config_files)} dataset configuration file(s):\n")
        for i, config_file in enumerate(config_files, 1):
            mod_time = datetime.fromtimestamp(config_file.stat().st_mtime)
            print(f"  {i}. {config_file.name}")
            print(f"     Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"     Path: {config_file}")
            print()
        
        return config_files
    
    def list_cached_datasets(self):
        """List locally cached datasets"""
        self.print_section("Cached Datasets")
        
        cached_files = list(Path(DATASET_CACHE_DIR).glob("*.pkl")) + \
                      list(Path(DATASET_CACHE_DIR).glob("*.parquet")) + \
                      list(Path(DATASET_CACHE_DIR).glob("*.csv"))
        
        if not cached_files:
            print("  No cached datasets found.")
            return []
        
        print(f"\n  Found {len(cached_files)} cached dataset(s):\n")
        for i, cache_file in enumerate(cached_files, 1):
            mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            print(f"  {i}. {cache_file.name}")
            print(f"     Size: {size_mb:.2f} MB")
            print(f"     Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        
        return cached_files
    
    def search_huggingface_datasets(self, query: str) -> List[Dict[str, Any]]:
        """Search Hugging Face for datasets"""
        self.print_section(f"Searching for datasets matching: '{query}'")
        
        try:
            # Search using Hugging Face API
            url = "https://huggingface.co/api/datasets"
            params = {
                "search": query,
                "sort": "downloads",
                "direction": -1,
                "limit": 50
            }
            
            print("  Querying Hugging Face API...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            datasets = response.json()
            
            # Filter for datasets that actually contain the search string
            filtered_datasets = [
                d for d in datasets 
                if query.lower() in d.get('id', '').lower() or 
                   query.lower() in (d.get('description', '') or '').lower()
            ]
            
            self.print_success(f"Found {len(filtered_datasets)} matching datasets")
            return filtered_datasets[:20]  # Limit to top 20
            
        except Exception as e:
            self.print_error(f"Failed to search Hugging Face: {str(e)}")
            return []
    
    def display_datasets_paginated(self, datasets: List[Dict[str, Any]]) -> Optional[str]:
        """Display datasets in a paginated, numbered list"""
        if not datasets:
            self.print_warning("No datasets found.")
            return None
        
        self.print_section("Available Datasets")
        
        for i, dataset in enumerate(datasets, 1):
            dataset_id = dataset.get('id', 'Unknown')
            downloads = dataset.get('downloads', 0)
            likes = dataset.get('likes', 0)
            
            print(f"\n  {Colors.BOLD}{i}. {dataset_id}{Colors.ENDC}")
            print(f"     Downloads: {downloads:,} | Likes: {likes}")
            
            # Get description
            description = dataset.get('description', 'No description available') or 'No description available'
            
            # Truncate long descriptions
            if len(description) > 100:
                description = description[:97] + "..."
            print(f"     {description}")
            
            # Show tags if available
            tags = dataset.get('tags', [])
            if tags:
                tag_str = ', '.join(tags[:5])
                if len(tags) > 5:
                    tag_str += f" (+{len(tags)-5} more)"
                print(f"     Tags: {tag_str}")
        
        print(f"\n{Colors.OKCYAN}Enter a number to select a dataset, 'b' to go back, or Ctrl+C to exit{Colors.ENDC}")
        
        while True:
            try:
                choice = input(f"\n{Colors.BOLD}Your choice: {Colors.ENDC}").strip()
                
                if choice.lower() == 'b':
                    return None
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(datasets):
                    selected = datasets[choice_num - 1]
                    return selected.get('id')
                else:
                    self.print_error(f"Please enter a number between 1 and {len(datasets)}")
                    
            except ValueError:
                self.print_error("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n")
                raise
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get detailed information about a dataset from HuggingFace"""
        try:
            url = f"https://huggingface.co/api/datasets/{dataset_name}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.print_warning(f"Could not fetch dataset info: {str(e)}")
            return {}
    
    def get_dataset_configs(self, dataset_name: str) -> List[str]:
        """Get available configurations for a dataset"""
        try:
            from datasets import get_dataset_config_names
            configs = get_dataset_config_names(dataset_name)
            return configs
        except ImportError:
            return []
        except Exception as e:
            self.print_warning(f"Could not fetch configs: {str(e)}")
            return []
    
    def get_dataset_parameters(self, dataset_name: str) -> Dict[str, Any]:
        """Get default parameters for loading a dataset"""
        default_params = {
            "cache_dir": {
                "value": DATASET_CACHE_DIR,
                "description": "Directory to cache downloaded datasets",
                "type": "str"
            },
            "split": {
                "value": "train",
                "description": "Dataset split to load (train, test, validation)",
                "type": "str",
                "options": ["train", "test", "validation", "all"]
            },
            "sample_size": {
                "value": None,
                "description": "Number of samples to load (None for all)",
                "type": "int",
                "range": [1, 1000000]
            },
            "save_format": {
                "value": "parquet",
                "description": "Format to save cached dataset",
                "type": "str",
                "options": ["parquet", "csv", "pickle"]
            },
            "use_duckdb": {
                "value": True,
                "description": "Use DuckDB for loading (faster for large datasets)",
                "type": "bool"
            }
        }
        
        return default_params
    
    def display_parameters(self, params: Dict[str, Any]):
        """Display parameters with their current values"""
        print(f"\n{Colors.HEADER}Current Parameters:{Colors.ENDC}\n")
        
        for i, (key, param_info) in enumerate(params.items(), 1):
            value = param_info['value']
            desc = param_info['description']
            param_type = param_info['type']
            
            print(f"  {Colors.BOLD}{i}. {key}{Colors.ENDC}")
            print(f"     Value: {Colors.OKGREEN}{value}{Colors.ENDC} ({param_type})")
            print(f"     {desc}")
            print()
    
    def configure_parameters(self, dataset_name: str) -> Dict[str, Any]:
        """Interactive parameter configuration"""
        params = self.get_dataset_parameters(dataset_name)
        
        self.print_section("Dataset Loading Configuration")
        
        print(f"\n{Colors.OKCYAN}Options:{Colors.ENDC}")
        print("  (L) Load existing configuration file")
        print("  (C) Create new configuration")
        print("  (D) Use default values")
        print("  (B) Go back")
        
        while True:
            try:
                choice = input(f"\n{Colors.BOLD}Your choice: {Colors.ENDC}").strip().upper()
                
                if choice == 'B':
                    return None
                elif choice == 'L':
                    loaded_params = self.load_config_file()
                    if loaded_params:
                        return loaded_params
                elif choice == 'C':
                    return self.create_new_configuration(params)
                elif choice == 'D':
                    self.display_parameters(params)
                    confirm = input(f"\n{Colors.OKCYAN}Use these default values? (y/n): {Colors.ENDC}").strip().lower()
                    if confirm == 'y':
                        return params
                else:
                    self.print_error("Invalid choice. Please enter L, C, D, or B.")
                    
            except KeyboardInterrupt:
                print("\n")
                raise
    
    def load_config_file(self) -> Optional[Dict[str, Any]]:
        """Load an existing configuration file"""
        self.print_section("Load Configuration File")
        
        print(f"\n{Colors.OKCYAN}Enter filename (in {CONFIG_DIR})")
        print(f"   or full filepath to load:{Colors.ENDC}")
        
        filepath = input(f"\n{Colors.BOLD}Path: {Colors.ENDC}").strip()
        
        if not filepath:
            self.print_warning("No file specified.")
            return None
        
        if not os.path.isabs(filepath):
            filepath = os.path.join(CONFIG_DIR, filepath)
        
        if not filepath.endswith(('.yaml', '.json', '.yml')):
            filepath += '.yaml'
        
        try:
            with open(filepath, 'r') as f:
                if filepath.endswith('.json'):
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)
            
            self.print_success(f"Configuration loaded from {filepath}")
            self.config_file_path = filepath
            return config.get('parameters', config)
            
        except FileNotFoundError:
            self.print_error(f"File not found: {filepath}")
            return None
        except Exception as e:
            self.print_error(f"Error loading configuration: {str(e)}")
            return None
    
    def create_new_configuration(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new configuration interactively"""
        self.print_section("Create New Configuration")
        
        self.display_parameters(params)
        
        print(f"\n{Colors.OKCYAN}Enter the numbers of parameters you want to change,")
        print(f"separated by commas (e.g., 1,3,5) or 'all' for all parameters:{Colors.ENDC}")
        
        selection = input(f"\n{Colors.BOLD}Parameters to edit: {Colors.ENDC}").strip()
        
        if not selection:
            return params
        
        if selection.lower() == 'all':
            indices = list(range(1, len(params) + 1))
        else:
            try:
                indices = [int(x.strip()) for x in selection.split(',')]
            except ValueError:
                self.print_error("Invalid input. Using default values.")
                return params
        
        param_items = list(params.items())
        
        for idx in indices:
            if 1 <= idx <= len(param_items):
                key, param_info = param_items[idx - 1]
                params[key] = self.edit_parameter(key, param_info)
        
        saved_path = self.save_configuration(params)
        if saved_path:
            self.config_file_path = saved_path
        
        return params
    
    def edit_parameter(self, key: str, param_info: Dict[str, Any]) -> Dict[str, Any]:
        """Edit a single parameter"""
        print(f"\n{Colors.BOLD}Parameter: {key}{Colors.ENDC}")
        print(f"Description: {param_info['description']}")
        print(f"Current value: {Colors.OKGREEN}{param_info['value']}{Colors.ENDC} ({param_info['type']})")
        
        if param_info['type'] == 'bool':
            print(f"{Colors.OKCYAN}Options: true, false{Colors.ENDC}")
        elif 'options' in param_info:
            print(f"{Colors.OKCYAN}Options: {', '.join(map(str, param_info['options']))}{Colors.ENDC}")
        elif 'range' in param_info:
            print(f"{Colors.OKCYAN}Range: {param_info['range'][0]} - {param_info['range'][1]}{Colors.ENDC}")
        
        new_value = input(f"\n{Colors.BOLD}New value (or press Enter to keep current): {Colors.ENDC}").strip()
        
        if not new_value:
            return param_info
        
        try:
            if param_info['type'] == 'int':
                if new_value.lower() == 'none':
                    param_info['value'] = None
                else:
                    param_info['value'] = int(new_value)
            elif param_info['type'] == 'bool':
                param_info['value'] = new_value.lower() in ('true', 'yes', '1', 't', 'y')
            else:
                param_info['value'] = new_value
            
            self.print_success(f"Updated {key} to {param_info['value']}")
            
        except ValueError:
            self.print_error(f"Invalid value for type {param_info['type']}. Keeping current value.")
        
        return param_info
    
    def save_configuration(self, params: Dict[str, Any]) -> Optional[str]:
        """Save configuration to file"""
        self.print_section("Save Configuration")
        
        timestamp = datetime.now().strftime("%m%d%y-%H%M")
        dataset_name_safe = self.selected_dataset.replace('/', '_').replace(' ', '_') if self.selected_dataset else "dataset"
        default_filename = f"dataset_{dataset_name_safe}-{timestamp}.yaml"
        
        print(f"\n{Colors.OKCYAN}Enter filename or full path to save configuration")
        print(f"Default: {default_filename}")
        print(f"Default location: {CONFIG_DIR}{Colors.ENDC}")
        
        filepath = input(f"\n{Colors.BOLD}Path (or press Enter for default): {Colors.ENDC}").strip()
        
        if not filepath:
            filepath = default_filename
        
        if not os.path.isabs(filepath):
            filepath = os.path.join(CONFIG_DIR, filepath)
        
        if not filepath.endswith(('.yaml', '.json', '.yml')):
            filepath += '.yaml'
        
        if os.path.exists(filepath):
            overwrite = input(f"\n{Colors.WARNING}File exists. Overwrite? (y/n): {Colors.ENDC}").strip().lower()
            if overwrite != 'y':
                self.print_warning("Save cancelled.")
                return None
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                save_data = {
                    "dataset_name": self.selected_dataset,
                    "created": datetime.now().isoformat(),
                    "parameters": {k: v['value'] for k, v in params.items()}
                }
                
                if filepath.endswith('.json'):
                    json.dump(save_data, f, indent=2)
                else:
                    yaml.dump(save_data, f, default_flow_style=False)
            
            self.print_success(f"Configuration saved to: {filepath}")
            return filepath
            
        except Exception as e:
            self.print_error(f"Failed to save configuration: {str(e)}")
            return None
    
    def load_dataset(self, dataset_name: str, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load the selected dataset with configured parameters"""
        self.print_section(f"Loading Dataset: {dataset_name}")
        
        param_values = {k: v['value'] for k, v in params.items()}
        
        # Check if already in cache
        cache_filename = f"{dataset_name.replace('/', '_')}_{param_values.get('split', 'train')}.{param_values.get('save_format', 'parquet')}"
        cache_path = os.path.join(DATASET_CACHE_DIR, cache_filename)
        
        if os.path.exists(cache_path):
            print(f"\n{Colors.OKCYAN}Found cached dataset at: {cache_path}{Colors.ENDC}")
            load_cached = input(f"{Colors.BOLD}Load from cache? (y/n): {Colors.ENDC}").strip().lower()
            
            if load_cached == 'y':
                try:
                    if cache_path.endswith('.parquet'):
                        df = pd.read_parquet(cache_path)
                    elif cache_path.endswith('.csv'):
                        df = pd.read_csv(cache_path)
                    elif cache_path.endswith('.pkl'):
                        df = pd.read_pickle(cache_path)
                    
                    self.print_success(f"Loaded {len(df)} rows from cache")
                    self.loaded_datasets[dataset_name] = df
                    return df
                except Exception as e:
                    self.print_warning(f"Failed to load cache: {str(e)}")
                    print("Will download fresh copy...")
        
        # Load from Hugging Face
        try:
            print(f"\n  Downloading dataset from Hugging Face...")
            print(f"  Dataset: {dataset_name}")
            print(f"  Split: {param_values.get('split', 'train')}")
            
            # Get token if available
            token = None
            if self.auth:
                token = self.auth.get_token()
                if token:
                    os.environ['HF_TOKEN'] = token
            
            df = None
            
            # Use DuckDB to load from HuggingFace
            if param_values.get('use_duckdb', True):
                hf_path = f"hf://datasets/{dataset_name}"
                
                # Try to detect parquet files
                try:
                    split = param_values.get('split', 'train')
                    
                    # Common patterns for HF dataset paths
                    possible_paths = [
                        f"{hf_path}/{split}/*.parquet",
                        f"{hf_path}/data/{split}/*.parquet",
                        f"{hf_path}/data/*.parquet",
                        f"{hf_path}/*.parquet",
                    ]
                    
                    for path in possible_paths:
                        try:
                            print(f"  Trying path: {path}")
                            df = duckdb.sql(f"SELECT * FROM '{path}';").df()
                            if df is not None and len(df) > 0:
                                self.print_success(f"Loaded via DuckDB from: {path}")
                                break
                        except Exception:
                            continue
                    
                    if df is None or len(df) == 0:
                        raise Exception("Could not find parquet files in any common location")
                    
                except Exception as e:
                    self.print_warning(f"DuckDB loading failed: {str(e)}")
                    print("  Falling back to datasets library...")
                    df = None  # Reset df to trigger fallback
            
            # Fallback to datasets library if DuckDB failed or was disabled
            if df is None:
                try:
                    from datasets import load_dataset as hf_load_dataset
                    
                    split = param_values.get('split', 'train')
                    print(f"  Loading with datasets library (split: {split})...")
                    
                    # Try loading with just dataset name and split first
                    try:
                        dataset = hf_load_dataset(dataset_name, split=split, token=token)
                        df = dataset.to_pandas()
                        self.print_success("Loaded via datasets library")
                    except Exception as e:
                        # Some datasets require a language/subset parameter
                        error_msg = str(e).lower()
                        if 'config' in error_msg or 'subset' in error_msg or 'language' in error_msg:
                            self.print_warning("Dataset requires additional configuration (language/subset)")
                            
                            # Try to get available configurations
                            configs = self.get_dataset_configs(dataset_name)
                            
                            print(f"\n{Colors.OKCYAN}This dataset requires a configuration parameter.{Colors.ENDC}")
                            if configs:
                                print(f"\n{Colors.BOLD}Available configurations:{Colors.ENDC}")
                                for i, config in enumerate(configs[:20], 1):  # Show first 20
                                    print(f"  {i}. {config}")
                                if len(configs) > 20:
                                    print(f"  ... and {len(configs) - 20} more")
                            else:
                                print(f"Common examples: 'en', 'fr', 'de', etc.")
                            
                            config_input = input(f"\n{Colors.BOLD}Enter configuration name (or 'skip' to cancel): {Colors.ENDC}").strip()
                            
                            if config_input and config_input.lower() != 'skip':
                                # Determine if input is a number or a name ('1', 'en', etc)
                                config_name = None

                                if config_input.isdigit():
                                    # User entered a number -- get config by index
                                    config_num = int(config_input)
                                    if configs and 1 <= config_num <= len(configs):
                                        config_name = configs[config_num - 1]
                                        print(f"    Selected: {config_name}")
                                    else:
                                        max_num = len(configs) if configs else 0
                                        raise Exception(f"Number {config_num} is out of range.  Please choose 1-{max_num}.")
                                else:
                                    # User entered a config name directly
                                    config_name = config_input

                                if config_name:
                                    try:
                                        print(f"  Loading with config: {config_name}...")
                                        dataset = hf_load_dataset(dataset_name, config_name, split=split, token=token)
                                        df = dataset.to_pandas()
                                        self.print_success(f"Loaded via datasets library (config: {config_name})")
                                    except Exception as e2:
                                        raise Exception(f"Failed with config '{config_name}': {str(e2)}")
                            else:
                                raise Exception("Dataset loading cancelled - configuration required")
                        else:
                            raise
                        
                except ImportError:
                    raise Exception("datasets library not installed. Run: pip install datasets")
                except Exception as e:
                    raise Exception(f"Failed to load with datasets library: {str(e)}")
            
            # Verify we loaded data
            if df is None or len(df) == 0:
                raise Exception("Dataset loaded but contains no data")
            
            # Apply sample size limit if specified
            sample_size = param_values.get('sample_size')
            if sample_size and len(df) > sample_size:
                print(f"  Sampling {sample_size} rows from {len(df)} total rows...")
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
            self.print_success(f"Dataset loaded successfully!")
            print(f"\n  Dataset: {dataset_name}")
            print(f"  Rows: {len(df):,}")
            print(f"  Columns: {', '.join(df.columns.tolist()[:10])}")
            if len(df.columns) > 10:
                print(f"           (+{len(df.columns)-10} more columns)")
            
            # Save to cache
            print(f"\n  Saving to cache...")
            try:
                save_format = param_values.get('save_format', 'parquet')
                if save_format == 'parquet':
                    df.to_parquet(cache_path, index=False)
                elif save_format == 'csv':
                    df.to_csv(cache_path, index=False)
                elif save_format == 'pickle':
                    df.to_pickle(cache_path)
                
                self.print_success(f"Cached to: {cache_path}")
            except Exception as e:
                self.print_warning(f"Failed to cache dataset: {str(e)}")
            
            self.loaded_datasets[dataset_name] = df
            return df
            
        except Exception as e:
            error_msg = str(e)
            
            # Check for authentication error
            if any(keyword in error_msg.lower() for keyword in ['401', '403', 'gated', 'authentication', 'unauthorized']):
                self.print_error("This dataset requires authentication")
                
                if self.auth:
                    print(f"\n{Colors.OKCYAN}Attempting to authenticate...{Colors.ENDC}")
                    if self.auth.ensure_authenticated():
                        self.print_success("Authentication successful! Please try loading again.")
                        return None
                    else:
                        self.print_error("Authentication failed")
                else:
                    self.print_error("Authentication module not available")
            else:
                self.print_error(f"Failed to load dataset: {error_msg}")
            
            return None
    
    def run(self) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """Main workflow"""
        self.print_header("DuckDB RAG - Dataset Configurator")
        
        try:
            # Show existing configs and cached datasets
            self.list_existing_configs()
            self.list_cached_datasets()
            
            # Authentication check
            if self.auth:
                print(f"\n{Colors.OKCYAN}Authentication status:{Colors.ENDC}")
                if self.auth.is_authenticated():
                    print(f"    {Colors.OKGREEN}✓ Authenticated{Colors.ENDC}")
                else:
                    print(f"    {Colors.WARNING}⚠ Not authenticated (required for gated datasets){Colors.ENDC}")
                
                login_choice = input(f"\n{Colors.BOLD}Login to HuggingFace now? (y/n, default=n): {Colors.ENDC}").strip().lower()
                if login_choice == 'y':
                    if not self.auth.interactive_login():
                        print(f"{Colors.WARNING}Continuing without authentication{Colors.ENDC}")
            
            # Step 1: Search for dataset
            while True:
                self.print_section("Dataset Search")
                print(f"\n{Colors.OKCYAN}Enter a search term to find datasets (e.g., 'legal', 'qa', 'thai'){Colors.ENDC}")
                print(f"{Colors.OKCYAN}Or type 'exit' to quit{Colors.ENDC}")
                
                search_query = input(f"\n{Colors.BOLD}Search: {Colors.ENDC}").strip()
                
                if not search_query or search_query.lower() == 'exit':
                    return None, None
                
                datasets = self.search_huggingface_datasets(search_query)
                
                if not datasets:
                    continue
                
                # Step 2: Select dataset
                selected_dataset = self.display_datasets_paginated(datasets)
                
                if selected_dataset:
                    self.selected_dataset = selected_dataset
                    break
            
            # Show dataset info
            dataset_info = self.get_dataset_info(self.selected_dataset)
            if dataset_info:
                print(f"\n{Colors.OKBLUE}Dataset Information:{Colors.ENDC}")
                if 'description' in dataset_info and dataset_info['description']:
                    desc = dataset_info['description']
                    if len(desc) > 200:
                        desc = desc[:197] + "..."
                    print(f"  {desc}")
            
            # Step 3: Configure parameters
            params = self.configure_parameters(self.selected_dataset)
            
            if params is None:
                return None, None
            
            # Step 4: Load dataset
            dataset = self.load_dataset(self.selected_dataset, params)
            
            if dataset is None:
                return None, None
            
            config = {
                "dataset_name": self.selected_dataset,
                "dataset": dataset,
                "parameters": {k: v['value'] for k, v in params.items()},
                "config_file": self.config_file_path
            }
            
            self.print_success("Dataset configuration complete!")
            
            return dataset, config
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.WARNING}Operation cancelled by user.{Colors.ENDC}")
            return None, None
        except Exception as e:
            self.print_error(f"Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

def main():
    """Standalone execution"""
    configurator = DatasetConfigurator()
    dataset, config = configurator.run()
    
    if dataset is not None and config:
        print(f"\n{Colors.OKGREEN}{'='*80}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}Dataset ready for use!{Colors.ENDC}")
        print(f"{Colors.OKGREEN}{'='*80}{Colors.ENDC}")
        return dataset, config
    else:
        print(f"\n{Colors.WARNING}No dataset configured.{Colors.ENDC}")
        return None, None

if __name__ == "__main__":
    main()
