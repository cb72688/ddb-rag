import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from select_proc import detect_proc


class DDBRagModel:
    # Singleton class to manage SentenceTransformer models and their settings
    _instance = None
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DDBRagModel, cls).__new__(cls)
            # Add other hyperparameters and settings here
            cls._instance.config = {
                "model_name": "jinaai/jina-embeddings-v3",
                "device": detect_proc(),
                "normalize_embeddings": True
            }
        return cls._instance

    def __init__(self, config_file=None):
        if config_file:
            self.load_config(config_file)
        self._load_model()

    def _load_model(self):
        # Loads or reloads the SentenceTransformer model based on the configuration and settings in the config_file
        model_name = self.config.get("model_name")
        device = self.config.get("device")
        normalize = self.config.get("normalize_embeddings")

        # Instantiate the model with current settings
        self._model = SentenceTransformer(
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize
        )
        print(f"Model '{model_name}' loaded successfully on device '{device}' with embedding normalization set to '{normalize}'")

    @property
    def model(self):
        # Provides global access to the SentenceTransformer model
        return self._model

    def update_settings(self, new_settings: dict):
        # Updates model settings and reloads the model to apply them
        self.config.update(new_settings)
        self._load_model()

    def save_config(self, filename: str):
        # Saves the current configuration to a JSON file
        config_path = Path(filename)
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"Configuration saved to {config_path}\nConfig:\n{self.config}")

    def load_config(self, filename: str):
        # Loads configuration from a JSON file and applies the settings
        config_path = Path(filename)
        if config_path.exists():
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            self.config.update(loaded_config)
            print(f"Configuration loaded from {config_path}")
            self._load_model()
        else:
            print(f"Warning: Configuration file '{filename}' not found")
