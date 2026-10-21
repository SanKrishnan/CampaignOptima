"""
CampaignOptima: Behavioral Feature Engineering & Preprocessing Pipeline
Engineers domain-driven features:
- Campaign_Intensity_Ratio
- Recency_Weight_Score
- Financial_Stability_Index
Handles categorical encoding, robust scaling, and SMOTE / class rebalancing.
"""

import logging
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from src.config import DATA_CONFIG, FEATURE_CONFIG, MODEL_CONFIG

logger = logging.getLogger(__name__)


class BehavioralFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for Advanced Marketing Analytics & Behavioral Modeling.
    Engineers domain features capturing customer financial liquidity and marketing contact fatigue.
    """

    def __init__(
        self,
        recency_decay_lambda: float = FEATURE_CONFIG.recency_decay_lambda,
        balance_upper_cap: float = FEATURE_CONFIG.balance_upper_cap,
        balance_lower_cap: float = FEATURE_CONFIG.balance_lower_cap
    ):
        self.recency_decay_lambda = recency_decay_lambda
        self.balance_upper_cap = balance_upper_cap
        self.balance_lower_cap = balance_lower_cap

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Derives sophisticated marketing features without data leakage.
        """
        X_out = X.copy()

        # 1. Campaign Intensity Ratio
        # Measures frequency of contacts during current campaign relative to historical touches.
        # Higher values signal intensive/aggressive telemarketing targeting.
        if "campaign" in X_out.columns and "previous" in X_out.columns:
            prev_safe = np.maximum(X_out["previous"].values, 0)
            X_out["Campaign_Intensity_Ratio"] = np.round(
                X_out["campaign"].values / (prev_safe + 1.0), 4
            )
        else:
            X_out["Campaign_Intensity_Ratio"] = 1.0

        # 2. Recency Weight Score
        # Transforms 'pdays' (-1 for never contacted, >0 for days elapsed) and 'previous' count.
        # Uses exponential decay recency weighting: recent contacts receive exponentially higher weight.
        if "pdays" in X_out.columns and "previous" in X_out.columns:
            pdays_val = X_out["pdays"].values
            prev_val = np.maximum(X_out["previous"].values, 0)

            # Cold lead (never contacted before): base score 0.0
            # Previously contacted: exp(-lambda * pdays) scaled by log(1 + previous)
            recency_decay = np.where(
                pdays_val < 0,
                0.0,
                np.exp(-self.recency_decay_lambda * np.clip(pdays_val, 0, 500))
            )
            engagement_multiplier = np.log1p(prev_val)
            X_out["Recency_Weight_Score"] = np.round(recency_decay * (1.0 + engagement_multiplier), 4)
        else:
            X_out["Recency_Weight_Score"] = 0.0

        # 3. Financial Stability Index (FSI)
        # Composite score blending winsorized account balance, housing loan liability, and personal loan liability.
        # Positive score denotes high liquidity/net-worth client with high capacity to deposit.
        if "balance" in X_out.columns:
            clipped_balance = np.clip(X_out["balance"].values, self.balance_lower_cap, self.balance_upper_cap)
            # Normalize clipped balance into [-1.0, 1.0]
            norm_balance = 2.0 * (clipped_balance - self.balance_lower_cap) / (self.balance_upper_cap - self.balance_lower_cap) - 1.0

            # Liabilities penalty
            has_housing = (X_out["housing"].astype(str).str.lower() == "yes").astype(float) if "housing" in X_out.columns else 0.0
            has_loan = (X_out["loan"].astype(str).str.lower() == "yes").astype(float) if "loan" in X_out.columns else 0.0
            has_default = (X_out["default"].astype(str).str.lower() == "yes").astype(float) if "default" in X_out.columns else 0.0

            fsi = norm_balance - (0.35 * has_housing) - (0.40 * has_loan) - (0.50 * has_default)
            X_out["Financial_Stability_Index"] = np.round(fsi, 4)
        else:
            X_out["Financial_Stability_Index"] = 0.0

        # 4. Supplementary Behavioral Features
        # Log-transformed duration (handling right-skew of call duration)
        if "duration" in X_out.columns:
            X_out["log_duration"] = np.round(np.log1p(np.maximum(X_out["duration"].values, 0)), 4)

        # Contact burnout flag: contacts >= 4 during this campaign
        if "campaign" in X_out.columns:
            X_out["is_high_frequency_contact"] = (X_out["campaign"].values >= 4).astype(int)

        return X_out


def build_preprocessor_pipeline(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> ColumnTransformer:
    """
    Constructs Scikit-Learn ColumnTransformer for numerical scaling and categorical encoding.
    """
    if numeric_features is None:
        numeric_features = DATA_CONFIG.numeric_columns + [
            "Campaign_Intensity_Ratio",
            "Recency_Weight_Score",
            "Financial_Stability_Index",
            "log_duration",
            "is_high_frequency_contact"
        ]

    if categorical_features is None:
        categorical_features = DATA_CONFIG.categorical_columns

    numeric_transformer = Pipeline(steps=[
        ("scaler", RobustScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


class CampaignOptimaFeaturePipeline:
    """
    Full End-to-End Feature Engineering Pipeline:
    Raw Data -> Behavioral Feature Engineering -> Encoding & Scaling -> Processed Matrix
    """

    def __init__(self):
        self.behavioral_engineer = BehavioralFeatureEngineer()
        self.preprocessor = None
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        X_feat = self.behavioral_engineer.fit_transform(X)
        self.preprocessor = build_preprocessor_pipeline()
        self.preprocessor.fit(X_feat)

        # Extract transformed column names
        num_names = self.preprocessor.transformers_[0][2]
        cat_encoder = self.preprocessor.transformers_[1][1].named_steps["onehot"]
        cat_raw_cols = self.preprocessor.transformers_[1][2]
        cat_names = list(cat_encoder.get_feature_names_out(cat_raw_cols))
        self.feature_names_out_ = list(num_names) + cat_names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_feat = self.behavioral_engineer.transform(X)
        X_mat = self.preprocessor.transform(X_feat)
        return pd.DataFrame(X_mat, columns=self.feature_names_out_, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def balance_with_smote(
        self,
        X_train_processed: pd.DataFrame,
        y_train: pd.Series,
        sampling_strategy: float = MODEL_CONFIG.smote_sampling_strategy,
        random_state: int = MODEL_CONFIG.random_state
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balances the severe class imbalance (~11% positive conversion rate) using SMOTE.
        Upsamples minority instances synthetically in the continuous feature space.
        """
        logger.info(f"Applying SMOTE class balancing. Pre-balance class distribution: {y_train.value_counts().to_dict()}")
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train_processed, y_train)
        X_res_df = pd.DataFrame(X_res, columns=self.feature_names_out_)
        logger.info(f"Post-SMOTE class distribution: {pd.Series(y_res).value_counts().to_dict()}")
        return X_res_df, pd.Series(y_res)
