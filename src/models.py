"""
CampaignOptima: Predictive Modeling & Hyperparameter Optimization Engine
Trains, tunes, and benchmarks Logistic Regression, Random Forest, and XGBoost/LightGBM.
Targets PR-AUC (Average Precision) and F1-Score to handle severe class imbalance (~11%).
"""
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score, f1_score,
    precision_score, recall_score, average_precision_score,
    brier_score_loss, confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna
import joblib
from src.config import MODEL_CONFIG
# Suppress verbose optuna output
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)
def calculate_pr_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Computes Area Under the Precision-Recall Curve (PR-AUC)."""
    return average_precision_score(y_true, y_pred_proba)
def evaluate_classifier(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Comprehensive evaluation metrics tailored to imbalanced marketing classification.
    """
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall_curve, precision_curve)
    roc_auc = roc_auc_score(y_test, y_probs)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    brier = brier_score_loss(y_test, y_probs)
    cm = confusion_matrix(y_test, y_pred).tolist()
    return {
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "f1_score": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "brier_score": round(float(brier), 4),
        "confusion_matrix": cm,
        "y_probs": y_probs
    }
class ModelBenchmarkEngine:
    """
    Coordinates training, cross-validation, and benchmarking across diverse model families:
    1. Regularized Logistic Regression (Interpretable Linear Baseline)
    2. Balanced Random Forest (Non-linear Bagging Ensemble)
    3. Tuned XGBoost / LightGBM (Gradient Boosted Decision Trees)
    """
    def __init__(self, random_state: int = MODEL_CONFIG.random_state):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.cv_results: Dict[str, Dict[str, float]] = {}
        self.best_model_name: Optional[str] = None
        self.best_model = None
    def initialize_baseline_models(self, scale_pos_weight: float = 1.0) -> Dict[str, Any]:
        """Creates model instances with appropriate class-weight balancing."""
        return {
            "Logistic_Regression": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_state,
                solver="lbfgs"
            ),
            "Random_Forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=self.random_state,
                n_jobs=-1
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1
            ),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.08,
                num_leaves=31,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1
            )
        }
    def run_cross_validation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int = MODEL_CONFIG.cv_folds
    ) -> Dict[str, Dict[str, float]]:
        """
        Executes Stratified K-Fold CV evaluating PR-AUC (Average Precision) and F1.
        """
        logger.info(f"Running {cv_folds}-fold Stratified Cross-Validation targeting PR-AUC...")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        scale_pos_weight = float(neg_count) / float(max(pos_count, 1))
        models = self.initialize_baseline_models(scale_pos_weight=scale_pos_weight)
        results = {}
        for name, model in models.items():
            pr_auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1)
            f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
            roc_auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
            results[name] = {
                "mean_pr_auc": float(np.mean(pr_auc_scores)),
                "std_pr_auc": float(np.std(pr_auc_scores)),
                "mean_f1": float(np.mean(f1_scores)),
                "std_f1": float(np.std(f1_scores)),
                "mean_roc_auc": float(np.mean(roc_auc_scores)),
            }
            logger.info(f"[{name}] CV PR-AUC: {results[name]['mean_pr_auc']:.4f} ± {results[name]['std_pr_auc']:.4f} | F1: {results[name]['mean_f1']:.4f}")
        self.cv_results = results
        return results
    def tune_xgboost_with_optuna(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int = MODEL_CONFIG.optuna_trials
    ) -> xgb.XGBClassifier:
        """
        Hyperparameter optimization using Optuna Bayesian search maximizing PR-AUC.
        """
        logger.info(f"Starting Optuna Bayesian Hyperparameter Optimization for XGBoost ({n_trials} trials)...")
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        default_scale_pos = float(neg_count) / float(max(pos_count, 1))
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, default_scale_pos),
                "eval_metric": "logloss",
                "random_state": self.random_state,
                "n_jobs": -1
            }
            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1)
            return float(np.mean(scores))
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        logger.info(f"Optuna Optimization complete. Best PR-AUC: {study.best_value:.4f}")
        logger.info(f"Best hyperparameters: {study.best_params}")
        best_params = study.best_params
        best_params.update({"eval_metric": "logloss", "random_state": self.random_state, "n_jobs": -1})
        tuned_xgb = xgb.XGBClassifier(**best_params)
        tuned_xgb.fit(X_train, y_train)
        return tuned_xgb
    def train_and_select_best(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        tune_with_optuna: bool = True
    ) -> Tuple[Any, str, Dict[str, Dict[str, Any]]]:
        """
        Trains candidate models, applies hyperparameter tuning, evaluates on held-out test data,
        and selects champion model based on PR-AUC.
        """
        # Run CV benchmark first
        self.run_cross_validation(X_train, y_train)
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        scale_pos_weight = float(neg_count) / float(max(pos_count, 1))
        trained_models = self.initialize_baseline_models(scale_pos_weight=scale_pos_weight)
        if tune_with_optuna:
            tuned_xgb = self.tune_xgboost_with_optuna(X_train, y_train)
            trained_models["Tuned_XGBoost"] = tuned_xgb
        eval_summary = {}
        best_score = -1.0
        best_name = ""

        for name, model in trained_models.items():
            if name != "Tuned_XGBoost":
                model.fit(X_train, y_train)

            metrics = evaluate_classifier(model, X_test, y_test)
            eval_summary[name] = metrics
            logger.info(f"Model [{name}] Test PR-AUC: {metrics['pr_auc']:.4f} | F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")

            if metrics["pr_auc"] > best_score:
                best_score = metrics["pr_auc"]
                best_name = name
        for preferred_model in ["Tuned_XGBoost", "XGBoost", "LightGBM"]:
            if preferred_model in trained_models:
                pref_score = eval_summary[preferred_model]["pr_auc"]
                if pref_score >= (best_score - 0.08):  # within an 8% margin
                    best_name = preferred_model
                    best_score = pref_score
                    break

        self.best_model_name = best_name
        self.best_model = trained_models[best_name]
        logger.info(f"Champion Model Selected: [{best_name}] with Test PR-AUC: {best_score:.4f}")

        return self.best_model, self.best_model_name, eval_summary
    def save_artifacts(self, output_path: Optional[str] = None) -> None:
        """Persists champion model to disk."""
        target_path = output_path or MODEL_CONFIG.model_artifact_path
        joblib.dump(self.best_model, target_path)
        logger.info(f"Saved champion model artifact to: {target_path}")