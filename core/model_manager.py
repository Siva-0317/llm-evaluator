"""
Model Manager - Handles model loading, inference, and registry
"""
import os
import time
import yaml
from pathlib import Path
from typing import List, Dict, Optional

class ModelManager:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.current_model = None
        self.current_model_name = None
        self.model_params = {}
        self.registry_path = Path("config/model_registry.yaml")

    def get_registry(self) -> List[Dict]:
        """Load model registry from YAML file"""
        if not self.registry_path.exists():
            return []

        with open(self.registry_path, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('models', []) if data else []

    def save_registry(self, registry: List[Dict]):
        """Save model registry to YAML file"""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            yaml.safe_dump({'models': registry}, f, sort_keys=False)

    def add_to_registry(self, name: str, path: str, model_type: str, tags: List[str]):
        """Add a model to the registry"""
        registry = self.get_registry()

        # Check if already exists
        for model in registry:
            if model['name'] == name:
                model['path'] = path
                model['type'] = model_type
                model['tags'] = tags
                self.save_registry(registry)
                return

        # Add new
        registry.append({
            'name': name,
            'path': path,
            'type': model_type,
            'tags': tags
        })
        self.save_registry(registry)

    def find_matching_models(self, task_desc: str, category: str = "auto-detect") -> List[Dict]:
        """Find models matching the task description"""
        registry = self.get_registry()

        # Extract keywords from task description
        keywords = set(task_desc.lower().split())

        # If category is specified and not auto-detect, use it
        if category != "auto-detect":
            keywords.add(category)

        matched = []
        for model in registry:
            # Check if any tag matches any keyword
            model_tags = set([tag.lower() for tag in model['tags']])
            model_type = model['type'].lower()

            # Calculate match score
            score = 0
            for keyword in keywords:
                if any(keyword in tag for tag in model_tags):
                    score += 2
                if keyword in model_type:
                    score += 1

            if score > 0:
                model_copy = model.copy()
                model_copy['match_score'] = score
                matched.append(model_copy)

        # Sort by score
        matched.sort(key=lambda x: x['match_score'], reverse=True)

        # Return top 3
        return matched[:3]

    def load_model(self, model_path: str, n_threads: int = 4, n_ctx: int = 2048,
                   temperature: float = 0.7, max_tokens: int = 512):
        """Load a GGUF model using llama-cpp-python"""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")

        self.model_params = {
            'n_threads': n_threads,
            'n_ctx': n_ctx,
            'temperature': temperature,
            'max_tokens': max_tokens
        }

        self.current_model = Llama(
            model_path=model_path,
            n_threads=n_threads,
            n_ctx=n_ctx,
            verbose=False
        )

        self.current_model_name = Path(model_path).stem

    def unload_model(self):
        """Unload the current model"""
        if self.current_model:
            del self.current_model
            self.current_model = None
            self.current_model_name = None

    def generate(self, messages: List[Dict], model_name: str = "current") -> Dict:
        """Generate response from messages"""
        if not self.current_model:
            raise RuntimeError("No model loaded")

        start_time = time.time()

        # Create chat completion
        response = self.current_model.create_chat_completion(
            messages=messages,
            temperature=self.model_params['temperature'],
            max_tokens=self.model_params['max_tokens']
        )

        latency = time.time() - start_time

        # Extract response
        text = response['choices'][0]['message']['content']
        tokens = response['usage']['completion_tokens']

        return {
            'text': text,
            'tokens': tokens,
            'latency': latency,
            'model_name': self.current_model_name
        }

    def generate_with_model(self, messages: List[Dict], model_info: Dict) -> Dict:
        """Generate response using a specific model from registry"""
        # Save current model state
        prev_model = self.current_model
        prev_name = self.current_model_name
        prev_params = self.model_params.copy()

        try:
            # Load the specified model
            self.load_model(
                model_info['path'],
                n_threads=prev_params.get('n_threads', 4),
                n_ctx=prev_params.get('n_ctx', 2048),
                temperature=prev_params.get('temperature', 0.7),
                max_tokens=prev_params.get('max_tokens', 512)
            )

            # Generate
            result = self.generate(messages)
            result['model_name'] = model_info['name']

            return result

        finally:
            # Restore previous model
            self.current_model = prev_model
            self.current_model_name = prev_name
            self.model_params = prev_params
