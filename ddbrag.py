#!/usr/bin/env python3
"""
DuckDB RAG System - Main Script
Integrated with Model Configurator for advanced model selection
"""

import duckdb
import pandas as pd
import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Import the model configurator
from ddb_sp_mdl_sel import ModelConfigurator, Colors
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
                        self.print_success("Model loaded: {model_name}")
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
    
    def load_or_create_dataframe(self, force_recreate: bool = False):
        """Load existing dataframe or create new one with embeddings"""
        pickle_file = 'dataframe.pkl'
        
        if not force_recreate and os.path.exists(pickle_file):
            self.print_section("Loading Existing Dataframe")
            try:
                self.dataframe = pd.read_pickle(pickle_file)
                self.print_success(f"Loaded dataframe with {len(self.dataframe)} rows")
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
        
        self.print_success(f"Dataframe saved with {len(df)} rows")
        return True
    
    def similarity_search(self, query: str, k: int = 5) -> pd.DataFrame:
        """Perform similarity search"""
        self.print_section(f"Similarity Search (top {k} results)")
        
        print(f"\nQuery: {query}")
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
            print(f"\n{Colors.OKCYAN}Question:{Colors.ENDC}")
            print(f"  {row['question']}")
            
            if 'answer' in row and pd.notna(row['answer']):
                print(f"\n{Colors.OKGREEN}Answer:{Colors.ENDC}")
                answer = str(row['answer'])
                if len(answer) > 300:
                    answer = answer[:297] + "..."
                print(f"  {answer}")
            
            print(f"\n{Colors.HEADER}{'-'*80}{Colors.ENDC}\n")
    
    def interactive_query(self):
        """Interactive query loop"""
        self.print_section("Interactive Query Mode")
        
        print(f"\n{Colors.OKCYAN}Enter your questions (or 'quit' to exit){Colors.ENDC}\n")
        
        while True:
            try:
                query = input(f"{Colors.BOLD}Query: {Colors.ENDC}").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                results = self.similarity_search(query)
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
            # Step 1: Choose setup method
            print(f"\n{Colors.OKCYAN}Setup Options:{Colors.ENDC}")
            print("1. Advanced Model Configurator (search, configure, save settings)")
            print("2. Quick Setup (use predefined models)")
            
            setup_choice = input(f"\n{Colors.BOLD}Choice (1-2, default=1): {Colors.ENDC}").strip()
            
            use_configurator = setup_choice != '2'
            
            # Step 2: Setup model
            if not self.setup_model(use_configurator):
                return
            
            # Step 3: Load/create dataframe
            print(f"\n{Colors.OKCYAN}Dataframe Options:{Colors.ENDC}")
            print("1. Load existing dataframe (if available)")
            print("2. Create new dataframe with embeddings")
            
            df_choice = input(f"\n{Colors.BOLD}Choice (1-2, default=1): {Colors.ENDC}").strip()
            
            force_recreate = df_choice == '2'
            
            if not self.load_or_create_dataframe(force_recreate):
                return
            
            # Step 4: Run queries
            print(f"\n{Colors.OKCYAN}Query Options:{Colors.ENDC}")
            print("1. Interactive mode (multiple queries)")
            print("2. Single query")
            
            query_choice = input(f"\n{Colors.BOLD}Choice (1-2, default=1): {Colors.ENDC}").strip()
            
            if query_choice == '2':
                # Single query
                query = input(f"\n{Colors.BOLD}Enter your query: {Colors.ENDC}").strip()
                if query:
                    results = self.similarity_search(query)
                    self.display_results(results)
            else:
                # Interactive mode
                self.interactive_query()
            
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
