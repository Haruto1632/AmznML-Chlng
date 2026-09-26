"""
run_experiment.py — Entry point for baseline experiment v1.

Member A — Baseline v1

Usage:
    python experiments/member_a/baseline_v1/run_experiment.py [--config path/to/config.yaml]
"""

import argparse
import sys
from pathlib import Path

import yaml

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import BaselinePipeline


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "experiment_config.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return config


def write_results_report(pipeline: BaselinePipeline, output_path: str):
    """Write experiment results to markdown."""
    results = pipeline.results
    config = pipeline.config
    
    report = f"""# Baseline Experiment v1 — Results

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Experiment:** {config['experiment']['name']}
**Description:** {config['experiment']['description']}

---

## Configuration

### Data
```yaml
{yaml.dump(config['data'], default_flow_style=False)}```

### Normalization
```yaml
{yaml.dump(config['normalization'], default_flow_style=False)}```

### Blocking
```yaml
{yaml.dump(config['blocking'], default_flow_style=False)}```

### Features
```yaml
{yaml.dump(config['features'], default_flow_style=False)}```

### Model
```yaml
{yaml.dump(config['model'], default_flow_style=False)}```

### Training
```yaml
{yaml.dump(config['training'], default_flow_style=False)}```

---

## Results

### Key Metrics

| Metric | Value |
|---|---|
| **Macro F_0.5** | **{results.get('macro_f05', 0):.4f}** |
| Macro Precision | {results.get('macro_precision', 0):.4f} |
| Macro Recall | {results.get('macro_recall', 0):.4f} |
| Singleton Accuracy | {results.get('singleton_accuracy', 0):.4f} |
| Non-Singleton F_0.5 | {results.get('non_singleton_f05', 0):.4f} |
| Blocking Recall | {results.get('blocking_recall', 0):.4f} |
| Best Threshold | {results.get('best_threshold', 0):.2f} |
| Runtime (seconds) | {results.get('runtime_seconds', 0):.1f} |

### Dataset Statistics

| Statistic | Value |
|---|---|
| Total S1 Entities (val) | {results.get('total_entities', 0):,} |
| Singletons | {results.get('singleton_count', 0):,} |
| Singleton Rate | {results.get('singleton_count', 0) / results.get('total_entities', 1):.2%} |

---

## Analysis

### Strengths
- Blocking recall: {results.get('blocking_recall', 0):.2%} of true matches survived blocking
- F_0.5 emphasizes precision (2× weight over recall)
- Conservative threshold: {results.get('best_threshold', 0):.2f} (reduces false positives)

### Limitations
- Multi-strategy blocking may have gaps for highly diverse name variations
- TF-IDF features sensitive to corpus size
- No cross-encoder or graph-based refinement yet

### Next Steps
1. Analyze false negatives: which true matches were missed by blocking?
2. Analyze false positives: which pairs incorrectly matched?
3. Consider additional blocking strategies (edit distance bands, embeddings)
4. Experiment with threshold per evidence level (high vs low confidence)
5. Investigate singleton false positives separately

---

## Files Generated

- `output/matching_results.tsv` — final predictions (validation set only)
- `output/candidate_pairs.tsv` — blocking candidates (validation set only)
- `RESULTS.md` — this report

---

_Experiment completed successfully._
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nResults report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiment v1")
    parser.add_argument('--config', type=str, default=None,
                        help="Path to config YAML (default: experiment_config.yaml)")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run pipeline
    pipeline = BaselinePipeline(config)
    results = pipeline.run()
    
    # Write results report
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = Path(__file__).parent / "RESULTS.md"
    write_results_report(pipeline, results_path)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {results_path}")
    print(f"Macro F_0.5: {results['macro_f05']:.4f}")
    print(f"Blocking recall: {results['blocking_recall']:.4f}")
    print(f"Runtime: {results['runtime_seconds']:.1f}s")


if __name__ == '__main__':
    # Import pandas here (after sys.path setup)
    import pandas as pd
    main()
