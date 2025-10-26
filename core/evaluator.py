"""
Evaluator - Handles evaluation sets and multi-model comparison
"""
import yaml
from typing import List, Dict

class Evaluator:
    def __init__(self, model_manager):
        self.model_manager = model_manager

    def run_on_multiple_models(self, messages: List[Dict], models: List[Dict]) -> List[Dict]:
        """Run the same prompt on multiple models and compare"""
        results = []

        for model_info in models:
            try:
                result = self.model_manager.generate_with_model(messages, model_info)
                results.append(result)
            except Exception as e:
                results.append({
                    'model_name': model_info['name'],
                    'text': f"Error: {str(e)}",
                    'tokens': 0,
                    'latency': 0,
                    'pass_fail': 'FAIL'
                })

        return results

    def load_eval_set(self, eval_file: str) -> List[Dict]:
        """Load evaluation set from YAML file"""
        with open(eval_file, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('cases', []) if data else []

    def run_eval_set(self, eval_file: str, models: List[Dict]) -> List[Dict]:
        """Run evaluation set across multiple models"""
        cases = self.load_eval_set(eval_file)
        all_results = []

        for model_info in models:
            model_results = []
            total_pass = 0
            total_latency = 0
            total_tokens = 0

            for case in cases:
                messages = [{"role": "user", "content": case['input']}]

                try:
                    result = self.model_manager.generate_with_model(messages, model_info)

                    # Simple pass/fail check
                    expected = case.get('expected', '').lower()
                    actual = result['text'].lower()

                    pass_fail = "PASS" if expected in actual or actual in expected else "FAIL"
                    if pass_fail == "PASS":
                        total_pass += 1

                    result['pass_fail'] = pass_fail
                    total_latency += result['latency']
                    total_tokens += result['tokens']

                    model_results.append(result)

                except Exception as e:
                    model_results.append({
                        'model_name': model_info['name'],
                        'text': f"Error: {str(e)}",
                        'tokens': 0,
                        'latency': 0,
                        'pass_fail': 'ERROR'
                    })

            # Create summary result
            summary = {
                'model_name': model_info['name'],
                'text': f"Passed {total_pass}/{len(cases)} cases",
                'tokens': total_tokens,
                'latency': total_latency,
                'pass_fail': f"{total_pass}/{len(cases)}",
                'rating': round((total_pass / len(cases)) * 10, 1) if cases else 0
            }

            all_results.append(summary)

        return all_results
