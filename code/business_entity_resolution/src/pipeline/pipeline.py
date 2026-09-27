"""Compatibility adapter from shared configuration to the production baseline."""
from argparse import Namespace
from pathlib import Path

from .production import train, infer, main


def run_pipeline(config: dict, mode: str = "test") -> None:
    """Run real training/inference; never substitute empty predictions for errors."""
    if mode not in ("train", "test", "infer"):
        raise ValueError("Use train to produce held-out evaluation metrics, or test/infer for inference")
    train_root = Path(config.get("data_dir_train", "student_resource/dataset/train")).parent
    test_root = Path(config.get("data_dir_test", "student_resource/dataset/test")).parent
    args = Namespace(
        data_dir=train_root if mode == "train" else test_root,
        work_dir=Path(config.get("work_dir", "output/work")),
        model_dir=Path(config.get("model_dir", "output/model")),
        output_dir=Path(config.get("output_dir", "output")),
        train_entities=config.get("train_entities", 30000), seed=config.get("seed", 42),
        batch_size=config.get("batch_size", 1000), bucket_limit=config.get("bucket_limit", 128),
        max_candidates=config.get("max_candidates", 24), max_keys=config.get("max_keys", 4),
        rounds=config.get("rounds", 300), threads=config.get("threads", 4),
    )
    (train if mode == "train" else infer)(args)
