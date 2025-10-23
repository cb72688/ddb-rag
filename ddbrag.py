#!/usr/bin/env python3
"""
DuckDB RAG System - Main Script
Integrated with Model Configurator and Dataset Configurator for advanced selection
"""

import duckdb
import pandas as pd
import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Import the configurators
from ddb_sp_mdl_sel import ModelConfigurator, Colors
from ddb_dataset_sel import DatasetConfigurator

try:
    from hf_auth import hfAuth
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

class DuckDBRAG:
    """Main RAG system class"""
    
    def __init__(self):
        self.model = None
        self.model_config = None
        self.dataframe = None
        self.datasets = {}  # Store multiple datasets
        self.current_dataset_name = None

        # Initialize authentication
        if AUTH_AVAILABLE:
            self.auth = hfAuth()
        else:
            self.auth = None
        
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
    
    def setup_model(self, use_configurator: bool = True):
        """Setup model using configurator or simple method"""
        if use_configurator:
            self.print_section("Using Advanced Model Configurator")
            configurator = ModelConfigurator()
            self.model, self.model_config = configurator.run()
            
            if self.model is None:
                self.print_warning("Model configuration cancelled. Exiting...")
                return False
        else:
            self.print_section("Using Quick Setup")
            self.model = self.quick_model_setup()
            
            if self.model is None:
                return False
        
        return True
    
    def quick_model_setup(self) -> SentenceTransformer:
        """Quick model setup without configurator"""
        print("\nQuick Model Options:")
        print("1. intfloat/multilingual-e5-large (Recommended for Thai)")
        print("2. sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        print("3. Custom model name")
        
        choice = input(f"\n{Colors.BOLD}Choice (1-3): {Colors.ENDC}").strip()
        
        model_map = {
            "1": "intfloat/multilingual-e5-large",
            "2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        }
        
        if choice in model_map:
            model_name = model_map[choice]
        elif choice == "3":
            model_name = input("Enter model name: ").strip()
        else:
            model_name = "intfloat/multilingual-e5-large"
        
        print(f"\nLoading {model_name}...")
        
        # Get token if available
        token = None
        if self.auth:
            token = self.auth.get_token()
            if token:
                os.environ['HF_TOKEN'] = token

        try:
            model = SentenceTransformer(model_name, device='cuda:0')
            self.print_success(f"Model loaded: {model_name}")
            return model
        except Exception as e:
            error_msg = str(e)

            # Check for authentication error
            if any(keyword in error_msg.lower() for keyword in ['401', '403', 'gated', 'authentication']):
                self.print_error("This model requires authentication")

                if self.auth and self.auth.ensure_authenticated():
                    self.print_success("Authentication successful, retrying...")
                    try:
                        token = self.auth.get_token()
                        os.environ['HF_TOKEN'] = token
                        model = SentenceTransformer(model_name, device='cuda:0', token=token)
                        self.print_success(f"Model loaded: {model_name}")
                        return model
                    except Exception as retry_error:
                        self.print_error(f"Failed: {retry_error}")
                else:
                    self.print_error("Authentication failed or not available")

            self.print_error(f"Failed to load model: {str(e)}")
            try:
                print("Trying CPU...")
                model = SentenceTransformer(model_name, device='cpu')
                self.print_success(f"Model loaded on CPU: {model_name}")
                return model
            except Exception as e2:
                self.print_error(f"Failed: {str(e2)}")
                return None
    
    def setup_dataset(self, use_configurator: bool = True):
        """Setup dataset using configurator or simple method"""
        if use_configurator:
            self.print_section("Using Advanced Dataset Configurator")
            configurator = DatasetConfigurator()
            dataset, config = configurator.run()
            
            if dataset is None:
                self.print_warning("Dataset configuration cancelled.")
                return False
            
            # Store the dataset
            dataset_name = config['dataset_name']
            self.datasets[dataset_name] = {
                'data': dataset,
                'config': config
            }
            self.current_dataset_name = dataset_name
            self.dataframe = dataset
            
            self.print_success(f"Dataset '{dataset_name}' loaded and set as current")
            return True
        else:
            # Use the legacy method
            return self.load_or_create_dataframe(force_recreate=False)
    
    def load_or_create_dataframe(self, force_recreate: bool = False):
        """Load existing dataframe or create new one with embeddings (legacy method)"""
        pickle_file = 'dataframe.pkl'
        
        if not force_recreate and os.path.exists(pickle_file):
            self.print_section("Loading Existing Dataframe")
            try:
                self.dataframe = pd.read_pickle(pickle_file)
                self.print_success(f"Loaded dataframe with {len(self.dataframe)} rows")
                self.current_dataset_name = "legacy_pickle"
                self.datasets['legacy_pickle'] = {
                    'data': self.dataframe,
                    'config': {'source': 'pickle_file'}
                }
                return True
            except Exception as e:
                self.print_error(f"Failed to load pickle: {str(e)}")
                self.print_warning("Will create new dataframe...")
        
        self.print_section("Creating New Dataframe with Embeddings")
        
        # Load dataset
        print("\nLoading dataset from Hugging Face...")
        try:
            df = duckdb.sql(
                "SELECT * FROM 'hf://datasets/airesearch/WangchanX-Legal-ThaiCCL-RAG/data/*.parquet';"
            ).df()
            self.print_success(f"Dataset loaded: {len(df)} rows")
        except Exception as e:
            self.print_error(f"Failed to load dataset: {str(e)}")
            return False
        
        # Create embeddings
        print(f"\nCreating embeddings (this may take several minutes)...")
        
        def encode_with_progress(text, idx):
            if idx % 100 == 0:
                print(f"  Processing row {idx}/{len(df)}...")
            try:
                return self.model.encode(text)
            except Exception as e:
                print(f"  Error at row {idx}: {str(e)}")
                return None
        
        df['embeddings'] = df.apply(
            lambda row: encode_with_progress(row['question'], row.name),
            axis=1
        )
        
        # Remove failed encodings
        initial_count = len(df)
        df = df[df['embeddings'].notna()]
        final_count = len(df)
        
        if initial_count != final_count:
            self.print_warning(f"{initial_count - final_count} rows failed encoding")
        
        # Save
        print("\nSaving dataframe...")
        df.to_pickle(pickle_file)
        self.dataframe = df
        self.current_dataset_name = "legacy_default"
        self.datasets['legacy_default'] = {
            'data': df,
            'config': {'source': 'WangchanX-Legal-ThaiCCL-RAG'}
        }
        
        self.print_success(f"Dataframe saved with {len(df)} rows")
        return True
    
    def list_datasets(self):
        """List all loaded datasets"""
        self.print_section("Loaded Datasets")
        
        if not self.datasets:
            print("  No datasets loaded.")
            return
        
        for i, (name, info) in enumerate(self.datasets.items(), 1):
            dataset = info['data']
            is_current = name == self.current_dataset_name
            marker = f"{Colors.OKGREEN}[CURRENT]{Colors.ENDC}" if is_current else ""
            
            print(f"\n  {Colors.BOLD}{i}. {name}{Colors.ENDC} {marker}")
            print(f"     Rows: {len(dataset):,}")
            print(f"     Columns: {', '.join(dataset.columns.tolist()[:5])}")
            if len(dataset.columns) > 5:
                print(f"              (+{len(dataset.columns)-5} more)")
    
    def select_dataset(self):
        """Allow user to select which dataset to use"""
        self.list_datasets()
        
        if not self.datasets:
            return False
        
        print(f"\n{Colors.OKCYAN}Enter dataset number to select, or 'c' to cancel:{Colors.ENDC}")
        
        choice = input(f"\n{Colors.BOLD}Your choice: {Colors.ENDC}").strip()
        
        if choice.lower() == 'c':
            return False
        
        try:
            idx = int(choice) - 1
            dataset_names = list(self.datasets.keys())
            
            if 0 <= idx < len(dataset_names):
                selected_name = dataset_names[idx]
                self.current_dataset_name = selected_name
                self.dataframe = self.datasets[selected_name]['data']
                self.print_success(f"Selected dataset: {selected_name}")
                return True
            else:
                self.print_error("Invalid selection")
                return False
        except ValueError:
            self.print_error("Invalid input")
            return False
    
    def create_embeddings_for_dataset(self, text_column: str = None):
        """Create embeddings for the current dataset"""
        if self.dataframe is None:
            self.print_error("No dataset loaded")
            return False
        
        if 'embeddings' in self.dataframe.columns:
            overwrite = input(f"\n{Colors.WARNING}Embeddings already exist. Overwrite? (y/n): {Colors.ENDC}").strip().lower()
            if overwrite != 'y':
                return True
        
        # Determine which column to embed
        if text_column is None:
            print(f"\n{Colors.OKCYAN}Available columns:{Colors.ENDC}")
            for i, col in enumerate(self.dataframe.columns, 1):
                print(f"  {i}. {col}")
            
            col_choice = input(f"\n{Colors.BOLD}Select column to embed (number or name): {Colors.ENDC}").strip()
            
            try:
                if col_choice.isdigit():
                    col_idx = int(col_choice) - 1
                    text_column = self.dataframe.columns[col_idx]
                else:
                    text_column = col_choice
            except (IndexError, ValueError):
                self.print_error("Invalid column selection")
                return False
        
        if text_column not in self.dataframe.columns:
            self.print_error(f"Column '{text_column}' not found")
            return False
        
        self.print_section(f"Creating Embeddings for '{text_column}'")
        
        def encode_with_progress(text, idx):
            if idx % 100 == 0:
                print(f"  Processing row {idx}/{len(self.dataframe)}...")
            try:
                return self.model.encode(str(text))
            except Exception as e:
                print(f"  Error at row {idx}: {str(e)}")
                return None
        
        print(f"\nEncoding {len(self.dataframe)} rows...")
        self.dataframe['embeddings'] = self.dataframe.apply(
            lambda row: encode_with_progress(row[text_column], row.name),
            axis=1
        )
        
        # Remove failed encodings
        initial_count = len(self.dataframe)
        self.dataframe = self.dataframe[self.dataframe['embeddings'].notna()]
        final_count = len(self.dataframe)
        
        if initial_count != final_count:
            self.print_warning(f"{initial_count - final_count} rows failed encoding")
        
        # Update in datasets dict
        if self.current_dataset_name:
            self.datasets[self.current_dataset_name]['data'] = self.dataframe
        
        self.print_success(f"Embeddings created for {final_count} rows")
        
        # Offer to save
        save_choice = input(f"\n{Colors.OKCYAN}Save dataset with embeddings? (y/n): {Colors.ENDC}").strip().lower()
        if save_choice == 'y':
            filename = f"{self.current_dataset_name}_embedded.pkl"
            self.dataframe.to_pickle(filename)
            self.print_success(f"Saved to {filename}")
        
        return True
    
    def similarity_search(self, query: str, k: int = 5) -> pd.DataFrame:
        """Perform similarity search"""
        self.print_section(f"Similarity Search (top {k} results)")
        
        if self.dataframe is None:
            self.print_error("No dataset loaded")
            return pd.DataFrame()
        
        if 'embeddings' not in self.dataframe.columns:
            self.print_error("Dataset does not have embeddings. Create embeddings first.")
            return pd.DataFrame()
        
        print(f"\nDataset: {self.current_dataset_name}")
        print(f"Query: {query}")
        print("Encoding query...")
        
        query_vector = self.model.encode(query)
        embedding_dim = self.model.get_sentence_embedding_dimension()
        
        self.print_success("Query encoded")
        
        # Register dataframe with DuckDB
        duckdb.register('db', self.dataframe)
        
        sql = f"""
            SELECT
                *,
                array_cosine_distance(
                    embeddings::float[{embedding_dim}],
                    {query_vector.tolist()}::float[{embedding_dim}]
                ) as distance
            FROM 'db'
            ORDER BY distance
            LIMIT {k}
        """
        
        print("Executing search...")
        results = duckdb.sql(sql).to_df()
        
        self.print_success(f"Found {len(results)} results")
        return results
    
    def display_results(self, results: pd.DataFrame):
        """Display search results in a formatted way"""
        print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'SEARCH RESULTS'.center(80)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")
        
        for idx, row in results.iterrows():
            print(f"{Colors.BOLD}Rank {idx + 1}{Colors.ENDC} (Distance: {row['distance']:.4f})")
            
            # Display all text columns (excluding embeddings and distance)
            for col in results.columns:
                if col not in ['embeddings', 'distance'] and pd.api.types.is_string_dtype(results[col]):
                    value = str(row[col])
                    if pd.notna(value) and value:
                        print(f"\n{Colors.OKCYAN}{col}:{Colors.ENDC}")
                        if len(value) > 300:
                            value = value[:297] + "..."
                        print(f"  {value}")
            
            print(f"\n{Colors.HEADER}{'-'*80}{Colors.ENDC}\n")
    
    def interactive_query(self):
        """Interactive query loop"""
        self.print_section("Interactive Query Mode")
        
        print(f"\n{Colors.OKCYAN}Commands:{Colors.ENDC}")
        print("  <query>  - Search the dataset")
        print("  /switch  - Switch to a different dataset")
        print("  /list    - List all loaded datasets")
        print("  /embed   - Create embeddings for current dataset")
        print("  /load    - Load a new dataset")
        print("  /quit    - Exit interactive mode")
        
        print(f"\n{Colors.OKCYAN}Current dataset: {self.current_dataset_name}{Colors.ENDC}\n")
        
        while True:
            try:
                query = input(f"{Colors.BOLD}Query: {Colors.ENDC}").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['/quit', '/exit', '/q']:
                    break
                elif query.lower() == '/switch':
                    self.select_dataset()
                    print(f"\n{Colors.OKCYAN}Current dataset: {self.current_dataset_name}{Colors.ENDC}\n")
                elif query.lower() == '/list':
                    self.list_datasets()
                elif query.lower() == '/embed':
                    self.create_embeddings_for_dataset()
                elif query.lower() == '/load':
                    self.print_warning("Please restart the application to load a new dataset")
                else:
                    results = self.similarity_search(query)
                    if len(results) > 0:
                        self.display_results(results)
                
            except KeyboardInterrupt:
                print("\n")
                break
            except Exception as e:
                self.print_error(f"Error: {str(e)}")
    
    def run(self):
        """Main execution flow"""
        self.print_header("DuckDB RAG System")
        
        try:
            # Step 1: Choose model setup method
            print(f"\n{Colors.OKCYAN}Model Setup Options:{Colors.ENDC}")
            print("1. Advanced Model Configurator (search, configure, save settings)")
            print("2. Quick Setup (use predefined models)")
            
            setup_choice = input(f"\n{Colors.BOLD}Choice (1-2, default=1): {Colors.ENDC}").strip()
            
            use_model_configurator = setup_choice != '2'
            
            # Step 2: Setup model
            if not self.setup_model(use_model_configurator):
                return
            
            # Step 3: Choose dataset setup method
            print(f"\n{Colors.OKCYAN}Dataset Setup Options:{Colors.ENDC}")
            print("1. Advanced Dataset Configurator (search HuggingFace datasets)")
            print("2. Legacy method (use default Thai legal dataset or local pickle)")
            
            dataset_choice = input(f"\n{Colors.BOLD}Choice (1-2, default=1): {Colors.ENDC}").strip()
            
            use_dataset_configurator = dataset_choice != '2'
            
            # Step 4: Setup dataset
            if not self.setup_dataset(use_dataset_configurator):
                # Allow continuing without dataset
                print(f"\n{Colors.WARNING}Continuing without dataset loaded{Colors.ENDC}")
            
            # Step 5: Check if embeddings exist
            if self.dataframe is not None and 'embeddings' not in self.dataframe.columns:
                print(f"\n{Colors.WARNING}Dataset does not have embeddings{Colors.ENDC}")
                create_emb = input(f"{Colors.BOLD}Create embeddings now? (y/n): {Colors.ENDC}").strip().lower()
                
                if create_emb == 'y':
                    self.create_embeddings_for_dataset()
            
            # Step 6: Query mode
            if self.dataframe is not None:
                print(f"\n{Colors.OKCYAN}Query Options:{Colors.ENDC}")
                print("1. Interactive mode (multiple queries with commands)")
                print("2. Single query")
                
                query_choice = input(f"\n{Colors.BOLD}Choice (1-2, default=1): {Colors.ENDC}").strip()
                
                if query_choice == '2':
                    # Single query
                    query = input(f"\n{Colors.BOLD}Enter your query: {Colors.ENDC}").strip()
                    if query:
                        results = self.similarity_search(query)
                        if len(results) > 0:
                            self.display_results(results)
                else:
                    # Interactive mode
                    self.interactive_query()
            else:
                self.print_warning("No dataset available for querying")
            
            self.print_success("Session complete!")
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.WARNING}Session interrupted by user.{Colors.ENDC}")
        except Exception as e:
            self.print_error(f"Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """Entry point"""
    rag = DuckDBRAG()
    rag.run()


if __name__ == "__main__":
    main()
