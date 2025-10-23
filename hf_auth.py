#!/usr/bin/env python3
# huggingface Authentication module -- Secure authentication for accessing gated repos

import os
import sys
import json
import getpass
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime
import logging

log_file = "logs/hf_auth.log"
logger = logging.getLogger("hf_auth")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file)
file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_format)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(file_format)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info(f"Logging Setup!  Currently logging to file '{log_file}' and outputting to terminal via StreamHandler")

# Third party imports
try:
    from huggingface_hub import login, logout, whoami, HfFolder
    from huggingface_hub.utils import HfHubHTTPError
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("huggingface_hub not installed.  Run: pip install huggingface_hub")

try:
    import keyring
    from keyring.errors import KeyringError
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    logger.warning("keyring not installed for secure storage.  Run: pip install keyring")

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILBLE = False
    logger.warning("cryptography not installed.  Run: pip install cryptography")

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    logger.warning("dotenv not available.  Run: pip install dotenv")

# Color codes for output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class hfAuth:
    ## Secure authentication manager for huggingface
    # Service name for keyring
    SERVICE_NAME = "huggingface_ddbrag"
    USERNAME = "default_user"

    # Environment variable names
    ENV_VARS = [
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
    ]

    def __init__(self):
        self.token = None
        self.user_info = None

        # Ensure directories exist
        self.config_dir = Path.home() / ".config" / "ddbrag"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Encryption key file
        self.key_file = self.config_dir / ".auth_key"

        # Load .env file if available
        if DOTENV_AVAILABLE:
            load_dotenv()

    def print_success(self, text: str):
        # Print success message
        logger.info(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

    def print_warning(self, text: str):
        # Print warning message
        logger.warning(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

    def print_error(self, text: str):
        # Print section header
        logger.info(f"\n{Colors.OKBLUE}{Colors.BOLD}>>> {text}{Colors.ENDC}")

    def print_section(self, text: str):
        """Print formatted section header"""
        print(f"\n{Colors.OKBLUE}{Colors.BOLD}>>> {text}{Colors.ENDC}")

    def _get_encryption_key(self) -> bytes:
        # Get or create encryption key for token storage
        if not CRYPTO_AVAILABLE:
            return None

        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            # Generate one key
            key = Fernet.generate_key()
            # Save with restricted permissions
            self.key_file.write_bytes(key)
            self.key_file.chmod(0x600) # Read/Write for owner only
            return key

    def _encrypt_token(self, token: str) -> str:
        # Encrypt token for secure storage
        if not CRYPTO_AVAILABLE:
            self.print_warning("Cryptography not available.  Token stored without encryption")
            return token
        try:
            key = self._get_encryption_key()
            f = Fernet(key)
            encrypted = f.encrypt(token.encode())
            return encrypted.decode()
        except Exception as e:
            self.print_error(f"Encryption failed: {e}")
            return token

    def _decrypt_token(self, encrypted_token: str) -> str:
        # Decrypt stored token
        if not CRYPTO_AVAILABLE:
            return encrypted_token

        try:
            key = self._get_encryption_key()
            f = Fernet(key)
            decrypted = f.decrypt(encrypted_token.encode())
            return decrypted.decode()
        except Exception as e:
            self.print_error(f"Decryption failed: {e}")
            return None

    def check_env_token(self) -> Optional[str]:
        # Check for token in environment variables
        for var in self.ENV_VARS:
            token = os.environ.get(var)
            if token:
                self.print_success(f"Found token in environemnt variable: {var}")
                return token
        return None

    def check_hf_folder_token(self) -> Optional[str]:
        # Check for token in huggingface cache folder
        if not HF_HUB_AVAILABLE:
            return None

        try:
            token = HfFolder.get_token()
            if token:
                self.print_success("Found token in huggingface cache")
                return token
        except Exception:
            pass
        return None

    def load_from_keyring(self) -> Optional[str]:
        # Load token from system keyring
        if not KEYRING_AVAILABLE:
            return None

        try:
            encrypted_token = keyring.get_password(self.SERVICE_NAME, self.USERNAME)
            if encrypted_token:
                token = self._decrypt_token(encrypted_token)
                if token:
                    self.print_success("Loaded token from secure keyring")
                    return token
        except KeyringError as e:
            self.print_warning(f"Could not access keyring: {e}")
        except Exception as e:
            self.print_warning(f"Error loading from keyring: {e}")
        return None

    def save_to_keyring(self, token: str) -> bool:
        # Save token to system keyring
        if not KEYRING_AVAILABLE:
            self.print_warning("Keyring not available.  Token not saved securely.")
            return False

        try:
            encrypted_token = self._encrypt_token(token)
            keyring.set_password(self.SERVICE_NAME, self.USERNAME, encrypted_token)
            self.print_success("Token saved to secure keyring")
            return True
        except KeyringError as e:
            self.print_error(f"Could not save to keyring: {e}")
            return False
        except Exception as e:
            self.print_error(f"Error saving to keyring: {e}")
            return False

    def validate_token(self, token: str) -> bool:
        # Validate token by checking with huggingface
        if not HF_HUB_AVAILABLE:
            self.print_error("huggingface_hub library not available")
            return False

        try:
            # Try to get user info with this token
            os.environ["HF_TOKEN"] = token
            user_info = whoami(token=token)
            self.user_info = user_info
            self.print_success("Authenticated as: {user_info.get('name', 'Unknown')}")
            return True
        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                self.print_error("Invalid token - authentication failed")
            else:
                self.print_error(f"Error validating token: {e}")
            return False
        except Exception as e:
            self.print_error(f"Error validating token: {e}")
            return False

    def prompt_for_token(self) -> Optional[str]:
        # Prompt user to enter token manually
        self.print_section("Enter huggingface token:")

        logger.info(f"\n{Colors.OKCYAN}You need a huggingface access token to access gated models.{Colors.ENDC}")
        logger.info(f"\n{Colors.BOLD}To get your token:{Colors.ENDC}")
        logger.info("1. Go to https://huggingface.co/settings/tokens")
        logger.info("2. Create a new token (read access is sufficient)")
        logger.info("3. Copy the token and paste it below\n")
        logger.info(f"\n{Colors.WARNING}Note: Token input will be hidden for security reasons{Colors.ENDC}\n")
        logger.info(f"\n{Colors.OKCYAN}Press Ctrl+C to cancel{Colors.ENDC}\n")

        try:
            token = getpass.getpass(f"{Colors.BOLD}Enter token: {Colors.ENDC}\n")
            return token.strip() if token else None
        except KeyboardInterrupt:
            print("\n")
            return None

    def login_huggingface(self, token: str, save: bool = True) -> bool:
        # Login to huggingface using token
        if not HF_HUB_AVAILABLE:
            self.print_error("huggingface_hub library not available")
            return False

        try:
            # Use huggingface_hub's login function
            login(token=token, add_to_git_credential=False)
            self.token = token

            # Validate and get user info
            if not self.validate_token(token):
                return False

            # Save to keyring if requested
            if save:
                self.save_to_keyring(token)

            self.print_success("Successfully logged in to huggingface!")
            return True

        except Exception as e:
            self.print_error(f"Login failed: {e}")
            return False

    def logout_huggingface(self) -> bool:
        # Logout from huggingface and clear credentials
        try:
            if HF_HUB_AVAILABLE:
                logout()

            # Clear from keyring
            if KEYRING_AVAILABLE:
                try:
                    keyring.delete_password(self.SERVICE_NAME, self.USERNAME)
                except Exception:
                    pass

            # Clear enviornment variable
            if "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]

            self.token = None
            self.user_info = None

            self.print_success("Logged out successfully!")
            return True
        except Exception as e:
            self.print_error(f"Logout failed: {e}")
            return False

    def is_authenticated(self) -> bool:
        # Check if user is currently authenticated
        # First check if there's a cached token for the user
        if self.token:
            return True

        # Try to find token from various sources
        token = (
            self.check_env_token() or
            self.check_hf_folder_token() or
            self.load_from_keyring()
        )

        if token:
            # Validate it
            if self.validate_token(token):
                self.token = token
                return True

        return False

    def ensure_authentication(self) -> bool:
        # Ensure user is authenticated; trigger prompt if needed
        if self.is_authenticated():
            return True

        self.print_warning("Authentication required for gated repositories")

        # Prompt for token
        token = self.prompt_for_token()
        if not token:
            return False

        # Try to login
        return self.login_huggingface(token, save=True)

    def interactive_login(self) -> bool:
        # Interactive login flow with options
        self.print_section("huggingface authentication")

        # Check if already authenticated
        if self.is_authenticated():
            logger.info(f"{Colors.OKGREEN}Already authenticated as: {self.user_info.get('name', 'Unknown')}{Colors.ENDC}")
            choice = input(f"\n{Colors.BOLD}Re-authenticate? (y/n): {Colors.ENDC}").strip().lower()
            if choice != 'y':
                return True

        logger.info(f"\n{Colors.OKCYAN}Authentication Options:{Colors.ENDC}")
        logger.info("1. Enter token manually")
        logger.info("2. Use environemnt variable")
        logger.info("3. Cancel")

        choice = input(f"\n{Colors.BOLD}Choice (1-3): {Colors.ENDC}").strip().lower()
        
        if choice == '1':
            token = self.prompt_for_token()
            if token:
                return self.login_huggingface(token, save=True)
            return False
        elif choice == '2':
            token = self.check_env_token()
            if token:
                return self.login_huggingface(token, save=True)
            else:
                self.print_error("No token found in environment variables")
                logger.info(f"\n{Colors.OKCYAN}Set one of these environment variables:{Colors.ENDC}")
                for var in self.ENV_VARS:
                    logger.info(f"   export {var}=your_token_here")
                return False
        else:
            return False

    def get_token(self) -> Optional[str]:
        # Get the current authentication token
        if not self.token:
            self.ensure_authentication()
        return self.token

    def handle_auth_error(self, error: Exception) -> bool:
        # Handle authentication error and attempt to fix
        self.print_error("Authentication error detected")

        # Check if it's a 401/403 error (unauthorized)
        if isinstance(error, HfHubHTTPError):
            if error.response.status_code in [401, 403]:
                logger.info(f"\n{Colors.WARNING}This model requires authentication or you don't have access{Colors.ENDC}")
                logger.info(f"\n{Colors.OKCYAN}Attempting to authenticate...{Colors.ENDC}")

                # Try to authenticate
                if self.ensure_authenticated():
                    self.print_success("Authentication successful.  Please retry your previously attempted action")
                    return True
                else:
                    self.print_error("Authentication failed")
                    logger.info(f"\n{Colors.WARNING}If this is a gated model, you may need to: {Colors.ENDC}\n")
                    logger.info("1. Go to the model page on huggingface.co")
                    logger.info("2. Accept the model's terms of use")
                    logger.info("3. Wait for access approval (if required)")
                    return False
        # Unknown error
        self.print_error(f"Error: {error}")
        return False

    def test_authentication():
        # Test authentication functionality
        auth = HuggingFaceAuth()

        logger.info(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
        logger.info(f"{Colors.HEADER}{Colors.BOLD}{'Hugging Face Authentication Test'.center(80)}{Colors.ENDC}\n")
        logger.info(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")

        # Check authentication status
        if auth.is_authenticated():
            logger.info(f"\n{Colors.OKGREEN}✓ Already authenticated!{Colors.ENDC}")
            logger.info(f"  User: {auth.user_info.get('name', 'Unkown')}")
            logger.info(f"  E-Mail: {auth.user_info.get('email', 'Unknown')}")
            return True
        else:
            logger.info(f"\n{Colors.WARNING}Not currently authenticated{Colors.ENDC}")

            # Try interactive login
            if auth.interactive_login():
                logger.info(f"\n{Colors.OKGREEN}✓ Authentication successful!{Colors.ENDC}")
                return True
            else:
                logger.info(f"\n{Colors.FAIL}✗ Authentication failed{Colors.ENDC}")
                return False

def main():
    # Main entrypoint for standalone usage
    import argparse
    parser = argparse.ArgumentParser(description="Hugging Face Authentication Manager")
    parser.add_argument('--login', action='store_true', help='Interactive login')
    parser.add_argument('--logout', action='store_true', help='Logout and clear credentials')
    parser.add_argument('--status', action='store_true', help='Check authentication status')
    parser.add_argument('--test', action='store_true', help='Test authentication')

    args = parser.parse_args()
