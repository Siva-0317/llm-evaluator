"""
Config Manager - Handles application configuration and state
"""
import yaml
from pathlib import Path

class ConfigManager:
    def __init__(self):
        self.config_path = Path("config/config.yaml")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> dict:
        """Load saved application state"""
        if not self.config_path.exists():
            return {}

        with open(self.config_path, 'r') as f:
            data = yaml.safe_load(f)
            return data if data else {}

    def save_state(self, state: dict):
        """Save application state"""
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(state, f, sort_keys=False)
