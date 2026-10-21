"""
CampaignOptima: AI-Driven Customer Conversion & Marketing Budget Optimization Engine
Package Configuration Module
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
for p in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class BusinessEconomicsConfig:
    """Marketing Campaign Unit Economics & Optimization Parameters."""
    cost_per_call: float = 15.0       # Direct telemarketing cost per contact ($)
    conversion_revenue: float = 300.0  # Net expected value of a successful term deposit ($)
    burnout_call_penalty: float = 5.0  # Friction / brand fatigue penalty per excess call (>3 calls)
    default_marketing_budget: float = 25000.0  # Default campaign budget ($)
    max_calls_per_customer: int = 5    # Contact frequency ceiling to prevent burnout


@dataclass
class DatasetConfig:
    """Dataset Schema & Delimiter Configurations."""
    raw_data_path: Path = RAW_DATA_DIR / "bank.csv"
    processed_train_path: Path = PROCESSED_DATA_DIR / "train_processed.parquet"
    processed_test_path: Path = PROCESSED_DATA_DIR / "test_processed.parquet"
    separator: str = ";"
    target_column: str = "y"
    target_positive_label: str = "yes"
    target_negative_label: str = "no"

    # Schema column definitions
    numeric_columns: List[str] = field(default_factory=lambda: [
        "age", "balance", "day", "duration", "campaign", "pdays", "previous"
    ])

    categorical_columns: List[str] = field(default_factory=lambda: [
        "job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"
    ])

    binary_indicator_columns: List[str] = field(default_factory=lambda: [
        "default", "housing", "loan"
    ])


@dataclass
class FeatureEngineeringConfig:
    """Behavioral Feature Engineering Parameters."""
    # Recency score half-life decay parameter (in days)
    recency_decay_lambda: float = 0.015
    # Balance normalization cap (winsorization at 99th percentile)
    balance_upper_cap: float = 25000.0
    balance_lower_cap: float = -1000.0


@dataclass
class ModelConfig:
    """Model Hyperparameter and Evaluation Settings."""
    random_state: int = 42
    test_size: float = 0.20
    cv_folds: int = 5
    primary_metric: str = "pr_auc"  # 'pr_auc' (Average Precision) or 'f1'
    optuna_trials: int = 30
    smote_sampling_strategy: float = 0.35  # Upsample minority class to 35% before class weighting
    model_artifact_path: Path = MODELS_DIR / "best_model.joblib"
    pipeline_artifact_path: Path = MODELS_DIR / "feature_pipeline.joblib"
    threshold_config_path: Path = MODELS_DIR / "optimal_threshold.json"


# Instantiate global singletons
BUSINESS_CONFIG = BusinessEconomicsConfig()
DATA_CONFIG = DatasetConfig()
FEATURE_CONFIG = FeatureEngineeringConfig()
MODEL_CONFIG = ModelConfig()
