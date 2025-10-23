#!/usr/bin/env python3
"""
Model Configurator for DuckDB RAG System
Handles model search, selection, and hyperparameter configuration
"""

import os
import sys
import json
import yaml
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import requests
from sentence_transformers import SentenceTransformer

# Import authentication module
try:
    from hf_auth import hfAuth
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    print("WARNING: hf_auth module not found.  Gated models may not be accessible")

# Configuration paths
USER_HOME = os.path.expanduser("~")
CONFIG_BASE = os.path.join(USER_HOME, ".config", "ddbrag")
CONFIG_DIR = os.path.join(CONFIG_BASE, "configs")
MODEL_DIR = os.path.join(CONFIG_BASE, "models")
LOG_DIR = "./logs/"
folders = [CONFIG_DIR, MODEL_DIR, LOG_DIR]

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

class ModelConfigurator:
    """Main class for model selection and configuration"""
    
    def __init__(self):
        self.selected_model = None
        self.model_config = {}
        self.config_file_path = None
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
        """List all existing configuration files"""
        self.print_section("Existing Configurations")
        
        config_files = list(Path(CONFIG_DIR).glob("*.yaml")) + list(Path(CONFIG_DIR).glob("*.json"))
        
        if not config_files:
            print("  No existing configuration files found.")
            return []
        
        print(f"\n  Found {len(config_files)} configuration file(s):\n")
        for i, config_file in enumerate(config_files, 1):
            # Get file modification time
            mod_time = datetime.fromtimestamp(config_file.stat().st_mtime)
            print(f"  {i}. {config_file.name}")
            print(f"     Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"     Path: {config_file}")
            print()
        
        return config_files
    
    def search_huggingface_models(self, query: str) -> List[Dict[str, Any]]:
        """Search Hugging Face for sentence-transformer models"""
        self.print_section(f"Searching for models matching: '{query}'")
        
        try:
            # Search using Hugging Face API
            url = "https://huggingface.co/api/models"
            params = {
                "search": query,
                "filter": "sentence-transformers",
                "sort": "downloads",
                "direction": -1,
                "limit": 50
            }
            
            print("  Querying Hugging Face API...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            models = response.json()
            
            # Filter for models that actually contain the search string
            filtered_models = [
                m for m in models 
                if query.lower() in m.get('id', '').lower() or 
                   query.lower() in m.get('modelId', '').lower()
            ]
            
            self.print_success(f"Found {len(filtered_models)} matching models")
            return filtered_models[:20]  # Limit to top 20
            
        except Exception as e:
            self.print_error(f"Failed to search Hugging Face: {str(e)}")
            return []
    
    def display_models_paginated(self, models: List[Dict[str, Any]]) -> Optional[str]:
        """Display models in a paginated, numbered list"""
        if not models:
            self.print_warning("No models found.")
            return None
        
        self.print_section("Available Models")
        
        for i, model in enumerate(models, 1):
            model_id = model.get('id', model.get('modelId', 'Unknown'))
            downloads = model.get('downloads', 0)
            likes = model.get('likes', 0)
            
            print(f"\n  {Colors.BOLD}{i}. {model_id}{Colors.ENDC}")
            print(f"     Downloads: {downloads:,} | Likes: {likes}")
            
            # Try to get description
            description = "No description available"
            if 'cardData' in model and model['cardData']:
                if isinstance(model['cardData'], dict):
                    description = model['cardData'].get('description', description)
            
            # Truncate long descriptions
            if len(description) > 100:
                description = description[:97] + "..."
            print(f"     {description}")
        
        print(f"\n{Colors.OKCYAN}Enter a number to select a model, 'b' to go back, or Ctrl+C to exit{Colors.ENDC}")
        
        while True:
            try:
                choice = input(f"\n{Colors.BOLD}Your choice: {Colors.ENDC}").strip()
                
                if choice.lower() == 'b':
                    return None
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(models):
                    selected = models[choice_num - 1]
                    return selected.get('id', selected.get('modelId'))
                else:
                    self.print_error(f"Please enter a number between 1 and {len(models)}")
                    
            except ValueError:
                self.print_error("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n")
                raise
    
    def get_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """Get default parameters for a model"""
        # Common sentence-transformer parameters
        default_params = {
            "device": {
                "value": "cuda:0",
                "description": "Device to run the model on (cuda:0, cuda:1, cpu)",
                "type": "str",
                "options": ["cuda:0", "cpu", "mps"]
            },
            "max_seq_length": {
                "value": 8192,
                "description": "Maximum sequence length for tokenization",
                "type": "int",
                "range": [64, 8192]
            },
            "batch_size": {
                "value": 64,
                "description": "Batch size for encoding",
                "type": "int",
                "range": [1, 256]
            },
            "normalize_embeddings": {
                "value": True,
                "description": "Whether to normalize embeddings to unit length",
                "type": "bool"
            },
            "trust_remote_code": {
                "value": True,
                "description": "Trust and execute remote code from model repository",
                "type": "bool"
            },
            "cache_folder": {
                "value": MODEL_DIR,
                "description": "Directory to cache downloaded models",
                "type": "str"
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
    
    def configure_parameters(self, model_name: str) -> Dict[str, Any]:
        """Interactive parameter configuration"""
        params = self.get_model_parameters(model_name)
        
        self.print_section("Parameter Configuration")
        
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
                    # If loading failed, continue to prompt
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
        
        # Check if it's a full path or just filename
        if not os.path.isabs(filepath):
            filepath = os.path.join(CONFIG_DIR, filepath)
        
        # Add extension if missing
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
            return config
            
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
        
        # Parse selection
        if selection.lower() == 'all':
            indices = list(range(1, len(params) + 1))
        else:
            try:
                indices = [int(x.strip()) for x in selection.split(',')]
            except ValueError:
                self.print_error("Invalid input. Using default values.")
                return params
        
        # Edit selected parameters
        param_items = list(params.items())
        
        for idx in indices:
            if 1 <= idx <= len(param_items):
                key, param_info = param_items[idx - 1]
                params[key] = self.edit_parameter(key, param_info)
        
        # Save configuration
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
        
        # Convert to appropriate type
        try:
            if param_info['type'] == 'int':
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
        
        # Generate default filename
        timestamp = datetime.now().strftime("%m%d%y-%H%M")
        model_name_safe = self.selected_model.replace('/', '_').replace(' ', '_') if self.selected_model else "model"
        default_filename = f"{model_name_safe}-{timestamp}.yaml"
        
        print(f"\n{Colors.OKCYAN}Enter filename or full path to save configuration")
        print(f"Default: {default_filename}")
        print(f"Default location: {CONFIG_DIR}{Colors.ENDC}")
        
        filepath = input(f"\n{Colors.BOLD}Path (or press Enter for default): {Colors.ENDC}").strip()
        
        if not filepath:
            filepath = default_filename
        
        # Check if it's a full path or just filename
        if not os.path.isabs(filepath):
            filepath = os.path.join(CONFIG_DIR, filepath)
        
        # Add extension if missing
        if not filepath.endswith(('.yaml', '.json', '.yml')):
            filepath += '.yaml'
        
        # Check for overwrite
        if os.path.exists(filepath):
            overwrite = input(f"\n{Colors.WARNING}File exists. Overwrite? (y/n): {Colors.ENDC}").strip().lower()
            if overwrite != 'y':
                self.print_warning("Save cancelled.")
                return None
        
        # Create directory if needed
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        except PermissionError:
            self.print_error("Permission denied. Attempting with sudo...")
            try:
                subprocess.run(['sudo', 'mkdir', '-p', os.path.dirname(filepath)], check=True)
            except Exception as e:
                self.print_error(f"Failed to create directory: {str(e)}")
                return self.save_configuration(params)  # Retry
        
        # Save configuration
        try:
            with open(filepath, 'w') as f:
                # Prepare data for saving
                save_data = {
                    "model_name": self.selected_model,
                    "created": datetime.now().isoformat(),
                    "parameters": {k: v['value'] for k, v in params.items()}
                }
                
                if filepath.endswith('.json'):
                    json.dump(save_data, f, indent=2)
                else:
                    yaml.dump(save_data, f, default_flow_style=False)
            
            self.print_success(f"Configuration saved to: {filepath}")
            return filepath
            
        except PermissionError:
            self.print_error("Permission denied.")
            use_sudo = input(f"{Colors.WARNING}Try with sudo? (y/n): {Colors.ENDC}").strip().lower()
            if use_sudo == 'y':
                # This won't work directly, but we inform the user
                self.print_error("Cannot use sudo for Python file operations.")
                self.print_warning("Please specify a path where you have write permissions.")
                return self.save_configuration(params)  # Retry
            return None
            
        except Exception as e:
            self.print_error(f"Failed to save configuration: {str(e)}")
            retry = input(f"\n{Colors.WARNING}Try again? (y/n): {Colors.ENDC}").strip().lower()
            if retry == 'y':
                return self.save_configuration(params)
            return None
    
    def load_model(self, model_name: str, params: Dict[str, Any]) -> Optional[SentenceTransformer]:
        """Load the selected model with configured parameters"""
        self.print_section(f"Loading Model: {model_name}")
        
        # Extract parameter values
        param_values = {k: v['value'] for k, v in params.items()}

        # Ensure authentication if auth module is available
        if self.auth and not self.auth.is_authenticated():
            print(f"\n{Colors.WARNING}Note: You may need authentication to access gated models{Colors.ENDC}")
            print(f"{Colors.OKCYAN}If the model is gated, authentication will be prompted automatically{Colors.ENDC}")

        try:
            print(f"\n  Downloading and initializing model...")
            print(f"  Device: {param_values.get('device', 'cpu')}")
            print(f"  Cache: {param_values.get('cache_folder', MODEL_DIR)}")

            # Get token if available
            token = None
            if self.auth:
                token = self.auth.get_token()
                if token:
                    # Set token in environment for sentence_transformers usage
                    os.environ['HF_TOKEN'] = token

            model = SentenceTransformer(
                model_name,
                device=param_values.get('device', 'cpu'),
                cache_folder=param_values.get('cache_folder', MODEL_DIR),
                trust_remote_code=param_values.get('trust_remote_code', True),
                token=token # Pass token to SentenceTransformers
            )
            
            # Set additional parameters
            if 'max_seq_length' in param_values:
                model.max_seq_length = param_values['max_seq_length']
            
            self.print_success(f"Model loaded successfully!")
            print(f"\n  Model: {model_name}")
            print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
            print(f"  Max sequence length: {model.max_seq_length}")
            
            return model
            
        except Exception as e:
            error_msg = str(e)

            # Check if it's an authentication error
            if any(keyword in error_msg.lower() for keyword in ['401', '403', 'gated', 'authentication', 'unauthorized']):
                self.print_error("This model requires authentication on huggingface")

                if self.auth:
                    print(f"\n{Colors.OKCYAN}Attempting to authenticate...{Colors.ENDC}")
                    if self.auth.ensure_authenticated():
                        self.print_success("Authentication successful!")
                        print(f"\n{Colors.BOLD}Please try loading the model again.{Colors.ENDC}")

                        # Retry once after authentication
                        try:
                            token = self.auth.get_token()
                            os.environ['HF_TOKEN'] = token

                            model = SentenceTransformer(
                                model_name,
                                device=param_values.get('device', 'cpu'),
                                cache_folder=param_values.get('cache_folder', MODEL_DIR),
                                trust_remote_code=param_values.get('trust_remote_code', True),
                                token=token
                            )

                            if 'max_seq_length' in param_values:
                                model.max_seq_length = param_values['max_seq_length']

                            self.print_success(f"Model loaded successfully after authentication!")
                            return model

                        except Exception as retry_error:
                            self.print_error(f"Failed again: {retry_error}")
                            return None
                    else:
                        self.print_error("Authentication failed")
                        print(f"\n{Colors.WARNING}For gated models, you may need to:{Colors.ENDC}\n")
                        print("1. Visit the model page on huggingface.co")
                        print("2. Accept the model's terms of use")
                        print("3. Wait for access approvial (if required)")
                        return None
                else:
                    self.print_error("Authentication module not available")
                    print(f"\n{Colors.WARNING}Install required packages:{Colors.ENDC}")
                    print("    pip install huggingface_hub keyring cryptography python-dotenv")
                    return None
            else:
                self.print_error(f"Failed to load model: {error_msg}")
                return None
    
    def run(self) -> Tuple[Optional[SentenceTransformer], Optional[Dict[str, Any]]]:
        """Main workflow"""
        self.print_header("DuckDB RAG - Model Configurator")
        
        try:
            # Show existing configs
            self.list_existing_configs()
            
            # Authentication check
            if self.auth:
                print(f"\n{Colors.OKCYAN}Authentication status:{Colors.ENDC}")
                if self.auth.is_authenticated():
                    print(f"    {Colors.OKGREEN}✓ Authenticated as {self.auth.user_info.get('name', 'Unknown')}{Colors.ENDC}")
                else:
                    print(f"    {Colors.WARNING}⚠ Not authenticated (required for gated models){Colors.ENDC}")

                # Offer to login now
                login_choice = input(f"\n{Colors.BOLD}Login to huggingface.co now? (y/n, default=n): {Colors.ENDC}").strip().lower()
                if login_choice == 'y':
                    if not self.auth.interactive_login():
                        print(f"{Colors.WARNING}Continuing without authentication{Colors.ENDC}")

            # Step 1: Search for model
            while True:
                self.print_section("Model Search")
                print(f"\n{Colors.OKCYAN}Enter a search term to find models (e.g., 'gemma', 'qwen', 'e5'){Colors.ENDC}")
                print(f"{Colors.OKCYAN}Or type 'exit' to quit{Colors.ENDC}")
                
                search_query = input(f"\n{Colors.BOLD}Search: {Colors.ENDC}").strip()
                
                if not search_query or search_query.lower() == 'exit':
                    return None, None
                
                # Search for models
                models = self.search_huggingface_models(search_query)
                
                if not models:
                    continue
                
                # Step 2: Select model
                selected_model = self.display_models_paginated(models)
                
                if selected_model:
                    self.selected_model = selected_model
                    break
            
            # Step 3: Configure parameters
            params = self.configure_parameters(self.selected_model)
            
            if params is None:
                return None, None
            
            # Step 4: Load model
            model = self.load_model(self.selected_model, params)
            
            if model is None:
                return None, None
            
            # Prepare config dict for return
            config = {
                "model_name": self.selected_model,
                "model": model,
                "parameters": {k: v['value'] for k, v in params.items()},
                "config_file": self.config_file_path
            }
            
            self.print_success("Configuration complete!")
            
            return model, config
            
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
    configurator = ModelConfigurator()
    model, config = configurator.run()
    
    if model and config:
        print(f"\n{Colors.OKGREEN}{'='*80}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}Model ready for use!{Colors.ENDC}")
        print(f"{Colors.OKGREEN}{'='*80}{Colors.ENDC}")
        return model, config
    else:
        print(f"\n{Colors.WARNING}No model configured.{Colors.ENDC}")
        return None, None

if __name__ == "__main__":
    main()
