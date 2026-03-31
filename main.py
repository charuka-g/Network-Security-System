from src.pipeline import TrainingPipeline
from src.logger import logging

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    artifact = pipeline.run_pipeline()
    print("\n=== Training Complete ===")
    print(f"  Train F1:  {artifact.train_metric.f1_score:.4f}")
    print(f"  Test  F1:  {artifact.test_metric.f1_score:.4f}")
    print(f"  Precision: {artifact.test_metric.precision_score:.4f}")
    print(f"  Recall:    {artifact.test_metric.recall_score:.4f}")
    print(f"  Model:     {artifact.trained_model_file_path}")
