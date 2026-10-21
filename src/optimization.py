"""
CampaignOptima: Decision Science & Operations Research Engine
Provides:
1. ThresholdOptimizer: Cost-benefit matrix & net profit curve optimization.
2. CampaignBudgetOptimizer: Linear Programming (MILP / LP) and Greedy Knapsack optimization
   using scipy.optimize to maximize conversions and ROI under budget and burnout constraints.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.config import BUSINESS_CONFIG

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Finds the profit-maximizing classification threshold using business unit economics
    instead of arbitrary 0.5 default.
    """

    def __init__(
        self,
        cost_per_call: float = BUSINESS_CONFIG.cost_per_call,
        conversion_revenue: float = BUSINESS_CONFIG.conversion_revenue
    ):
        self.cost_per_call = cost_per_call
        self.conversion_revenue = conversion_revenue
        self.theoretical_breakeven_threshold = cost_per_call / conversion_revenue

    def calculate_campaign_profit(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        threshold: float
    ) -> Dict[str, float]:
        """
        Calculates financial outcomes on empirical campaign observations at a given threshold.
        """
        # Call customer if predicted probability >= threshold
        calls_placed = (y_probs >= threshold).astype(int)
        total_calls = int(np.sum(calls_placed))

        # True conversions among customers called
        conversions = int(np.sum(calls_placed * y_true))

        total_spend = total_calls * self.cost_per_call
        gross_revenue = conversions * self.conversion_revenue
        net_profit = gross_revenue - total_spend
        roi = (net_profit / total_spend * 100.0) if total_spend > 0 else 0.0
        conversion_rate = (conversions / total_calls * 100.0) if total_calls > 0 else 0.0

        return {
            "threshold": round(float(threshold), 3),
            "total_calls": total_calls,
            "conversions": conversions,
            "total_spend": round(total_spend, 2),
            "gross_revenue": round(gross_revenue, 2),
            "net_profit": round(net_profit, 2),
            "roi_percent": round(roi, 2),
            "campaign_conversion_rate": round(conversion_rate, 2)
        }

    def compute_profit_curve(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        threshold_steps: int = 100
    ) -> pd.DataFrame:
        """
        Evaluates net profit across thresholds in [0.01, 0.99] to find optimal operating point.
        """
        thresholds = np.linspace(0.01, 0.95, threshold_steps)
        records = [self.calculate_campaign_profit(y_true, y_probs, t) for t in thresholds]
        df_curve = pd.DataFrame(records)
        return df_curve

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray
    ) -> Dict[str, Any]:
        """
        Identifies the optimal decision boundary tau* that maximizes total net campaign profit.
        """
        df_curve = self.compute_profit_curve(y_true, y_probs)
        best_row = df_curve.loc[df_curve["net_profit"].idxmax()]
        default_row = df_curve.iloc[(df_curve["threshold"] - 0.50).abs().argsort()[:1]].iloc[0]

        profit_uplift = best_row["net_profit"] - default_row["net_profit"]
        profit_uplift_pct = (profit_uplift / max(1.0, default_row["net_profit"])) * 100.0

        logger.info(f"Optimal Threshold: {best_row['threshold']:.3f} | Net Profit: ${best_row['net_profit']:,.2f}")
        logger.info(f"Default (0.50) Profit: ${default_row['net_profit']:,.2f} | Uplift: +${profit_uplift:,.2f} ({profit_uplift_pct:.1f}%)")

        return {
            "optimal_threshold": float(best_row["threshold"]),
            "optimal_metrics": best_row.to_dict(),
            "default_0_5_metrics": default_row.to_dict(),
            "profit_uplift": round(float(profit_uplift), 2),
            "profit_uplift_pct": round(float(profit_uplift_pct), 2),
            "profit_curve_df": df_curve,
            "theoretical_breakeven_threshold": round(self.theoretical_breakeven_threshold, 3)
        }


class CampaignBudgetOptimizer:
    """
    Operations Research & Knapsack Optimization Engine.
    Uses Linear Programming (scipy.optimize.linprog) and Greedy Knapsack heuristic
    to allocate limited marketing dollars / call capacity across customer cohorts.
    """

    def __init__(
        self,
        cost_per_call: float = BUSINESS_CONFIG.cost_per_call,
        conversion_revenue: float = BUSINESS_CONFIG.conversion_revenue,
        burnout_call_penalty: float = BUSINESS_CONFIG.burnout_call_penalty
    ):
        self.cost_per_call = cost_per_call
        self.conversion_revenue = conversion_revenue
        self.burnout_call_penalty = burnout_call_penalty

    def solve_budget_allocation_lp(
        self,
        customer_df: pd.DataFrame,
        probabilities: np.ndarray,
        budget_limit: float,
        max_call_capacity: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Linear Programming / Knapsack formulation solved with scipy.optimize:
        Maximize Expected Net Profit:
            sum_i x_i * (p_i * V_conv - C_call)
        Subject to:
            sum_i x_i * C_call <= Budget
            sum_i x_i <= Max_Calls (optional fatigue ceiling)
            0 <= x_i <= 1 (binary decision variable relaxed or integer solved)
        """
        n_customers = len(probabilities)
        if n_customers == 0:
            raise ValueError("No customers supplied for budget optimization.")

        # Objective: linprog minimizes c^T * x. To maximize, negate coefficients.
        # Net expected profit per contact = p_i * V_conv - C_call
        expected_profit = (probabilities * self.conversion_revenue) - self.cost_per_call
        c = -expected_profit

        # Inequality constraint matrix A_ub * x <= b_ub
        A_ub = []
        b_ub = []

        # Constraint 1: Total Dollar Budget
        # sum x_i * cost_per_call <= budget_limit
        A_ub.append(np.full(n_customers, self.cost_per_call))
        b_ub.append(budget_limit)

        # Constraint 2: Total Call Capacity (if specified)
        if max_call_capacity is not None and max_call_capacity > 0:
            A_ub.append(np.ones(n_customers))
            b_ub.append(max_call_capacity)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # Variable bounds: 0 <= x_i <= 1
        bounds = [(0, 1) for _ in range(n_customers)]

        # Solve using HiGHS solver
        res = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs"
        )

        if not res.success:
            logger.warning(f"Linear program solver finished with status: {res.message}. Falling back to greedy knapsack.")
            return self.solve_greedy_knapsack(customer_df, probabilities, budget_limit, max_call_capacity)

        # Continuous decision vector x in [0, 1] rounded to binary
        decision_vector = (res.x >= 0.5).astype(int)

        return self._format_optimization_results(
            customer_df, probabilities, decision_vector, budget_limit, method="Linear Programming (HiGHS)"
        )

    def solve_greedy_knapsack(
        self,
        customer_df: pd.DataFrame,
        probabilities: np.ndarray,
        budget_limit: float,
        max_call_capacity: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Greedy Marginal Efficiency Algorithm (Knapsack Heuristic):
        Sorts prospects by Marginal Return per Dollar:
            Efficiency_i = (p_i * V_conv - C_call) / C_call
        Picks prospects with positive return until budget or call quota is depleted.
        """
        n_customers = len(probabilities)
        expected_profit = (probabilities * self.conversion_revenue) - self.cost_per_call
        efficiency = expected_profit / max(1e-5, self.cost_per_call)

        # Order by efficiency descending
        sorted_indices = np.argsort(-efficiency)

        decision_vector = np.zeros(n_customers, dtype=int)
        current_spend = 0.0
        current_calls = 0

        max_calls = max_call_capacity if max_call_capacity else n_customers

        for idx in sorted_indices:
            # Only contact prospects with expected positive return
            if expected_profit[idx] <= 0:
                break
            if current_spend + self.cost_per_call > budget_limit:
                break
            if current_calls >= max_calls:
                break

            decision_vector[idx] = 1
            current_spend += self.cost_per_call
            current_calls += 1

        return self._format_optimization_results(
            customer_df, probabilities, decision_vector, budget_limit, method="Greedy Marginal Knapsack"
        )

    def _format_optimization_results(
        self,
        customer_df: pd.DataFrame,
        probabilities: np.ndarray,
        decision_vector: np.ndarray,
        budget_limit: float,
        method: str
    ) -> Dict[str, Any]:
        """Summarizes OR optimization outcomes and customer segment allocations."""
        selected_mask = (decision_vector == 1)
        total_selected = int(np.sum(selected_mask))

        total_spend = total_selected * self.cost_per_call
        expected_conversions = float(np.sum(probabilities[selected_mask]))
        expected_gross_revenue = expected_conversions * self.conversion_revenue
        expected_net_profit = expected_gross_revenue - total_spend
        roi_percent = (expected_net_profit / total_spend * 100.0) if total_spend > 0 else 0.0
        budget_utilization = (total_spend / budget_limit * 100.0) if budget_limit > 0 else 0.0

        # Construct targeted call sheet
        targeted_df = customer_df.copy()
        targeted_df["predicted_conversion_prob"] = np.round(probabilities, 4)
        targeted_df["expected_net_profit"] = np.round((probabilities * self.conversion_revenue) - self.cost_per_call, 2)
        targeted_df["is_targeted"] = decision_vector
        targeted_df = targeted_df.sort_values(by="predicted_conversion_prob", ascending=False)

        # Demographic targeting breakdown
        selected_prospects = targeted_df[targeted_df["is_targeted"] == 1]

        job_breakdown = selected_prospects["job"].value_counts().to_dict() if "job" in selected_prospects.columns else {}
        education_breakdown = selected_prospects["education"].value_counts().to_dict() if "education" in selected_prospects.columns else {}

        return {
            "optimization_method": method,
            "budget_limit": budget_limit,
            "total_spend": round(total_spend, 2),
            "budget_utilization_percent": round(budget_utilization, 2),
            "total_contacts_selected": total_selected,
            "expected_conversions": round(expected_conversions, 1),
            "expected_gross_revenue": round(expected_gross_revenue, 2),
            "expected_net_profit": round(expected_net_profit, 2),
            "expected_roi_percent": round(roi_percent, 2),
            "targeted_customers_df": targeted_df,
            "job_targeting_breakdown": job_breakdown,
            "education_targeting_breakdown": education_breakdown
        }
