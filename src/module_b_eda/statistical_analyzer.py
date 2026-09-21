"""
NeuroVision AI: Module B - Statistical Analyzer
Performs comparative clinical hypothesis testing (Mann-Whitney U) between HGG and LGG,
analyzes clinical covariate correlations (Age, Survival), and exports cohort-wide summaries.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from src.utils.logger import setup_logger

logger = setup_logger("StatisticalAnalyzer")


class StatisticalAnalyzer:
    """
    Evaluates volumetric and clinical differences across glioma grades.
    Supplies the statistical evidence base for Agent 2 (Clinical Decision Agent) RAG reasoning.
    """

    VOLUMETRIC_METRICS = [
        "wt_volume_cm3",
        "tc_volume_cm3",
        "et_volume_cm3",
        "ed_volume_cm3",
        "ncr_volume_cm3",
        "et_to_wt_ratio",
        "tc_to_wt_ratio",
        "ed_to_wt_ratio",
    ]

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    @staticmethod
    def _compute_iqr(series: pd.Series) -> float:
        clean = series.dropna()
        if len(clean) == 0:
            return 0.0
        q75, q25 = np.percentile(clean, [75, 25])
        return round(float(q75 - q25), 3)

    def compare_grades(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Runs non-parametric Mann-Whitney U tests comparing HGG vs LGG for each volumetric feature.
        """
        results: Dict[str, Dict[str, Any]] = {}
        hgg_df = df[df["grade"] == "HGG"]
        lgg_df = df[df["grade"] == "LGG"]

        for metric in self.VOLUMETRIC_METRICS:
            if metric not in df.columns:
                continue

            hgg_vals = hgg_df[metric].dropna()
            lgg_vals = lgg_df[metric].dropna()

            if len(hgg_vals) > 0 and len(lgg_vals) > 0:
                u_stat, p_val = stats.mannwhitneyu(hgg_vals, lgg_vals, alternative="two-sided")
                # Rank-Biserial correlation: r = 1 - (2U / (n1 * n2))
                n1, n2 = len(hgg_vals), len(lgg_vals)
                rank_biserial = round(1.0 - (2.0 * u_stat) / (n1 * n2), 4)

                results[metric] = {
                    "hgg_mean": round(float(hgg_vals.mean()), 3),
                    "hgg_std": round(float(hgg_vals.std()), 3),
                    "hgg_median": round(float(hgg_vals.median()), 3),
                    "hgg_iqr": self._compute_iqr(hgg_vals),
                    "lgg_mean": round(float(lgg_vals.mean()), 3),
                    "lgg_std": round(float(lgg_vals.std()), 3),
                    "lgg_median": round(float(lgg_vals.median()), 3),
                    "lgg_iqr": self._compute_iqr(lgg_vals),
                    "mann_whitney_u": float(u_stat),
                    "p_value": float(p_val),
                    "rank_biserial_r": rank_biserial,
                    "is_significant": bool(p_val < 0.05),
                }

        return results

    def correlate_clinical_covariates(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Computes Spearman rank correlations between tumor volumes and patient age/survival.
        """
        correlations: Dict[str, Dict[str, float]] = {}

        # Age correlation
        if "age" in df.columns:
            age_valid = df.dropna(subset=["age"])
            for metric in self.VOLUMETRIC_METRICS:
                if metric in age_valid.columns:
                    rho, p_val = stats.spearmanr(age_valid["age"], age_valid[metric])
                    correlations[f"age_vs_{metric}"] = {
                        "spearman_rho": round(float(rho), 4),
                        "p_value": float(p_val),
                    }

        # Survival correlation
        if "survival_days" in df.columns:
            # Coerce survival_days to float (some entries can be empty or strings)
            surv_numeric = pd.to_numeric(df["survival_days"], errors="coerce")
            surv_valid = df.loc[surv_numeric.notna()].copy()
            surv_valid["survival_numeric"] = surv_numeric.dropna()

            for metric in self.VOLUMETRIC_METRICS:
                if metric in surv_valid.columns:
                    rho, p_val = stats.spearmanr(surv_valid["survival_numeric"], surv_valid[metric])
                    correlations[f"survival_vs_{metric}"] = {
                        "spearman_rho": round(float(rho), 4),
                        "p_value": float(p_val),
                    }

        return correlations

    def build_dataset_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds a comprehensive tabular summary table (dataset_summary.csv) comparing HGG, LGG, and Cohort.
        """
        total_n = len(df)
        hgg_df = df[df["grade"] == "HGG"]
        lgg_df = df[df["grade"] == "LGG"]
        n_hgg = len(hgg_df)
        n_lgg = len(lgg_df)

        comparison = self.compare_grades(df)

        rows: List[Dict[str, Any]] = [
            {
                "Parameter": "Cohort Patient Count",
                "Cohort_Total": f"{total_n} (100.0%)",
                "HGG": f"{n_hgg} ({n_hgg / total_n * 100:.1f}%)",
                "LGG": f"{n_lgg} ({n_lgg / total_n * 100:.1f}%)",
                "P_Value": "N/A",
                "Significance": "N/A",
            }
        ]

        metric_display_names = {
            "wt_volume_cm3": "Whole Tumor Volume (WT, cm³)",
            "tc_volume_cm3": "Tumor Core Volume (TC, cm³)",
            "et_volume_cm3": "Enhancing Tumor Volume (ET, cm³)",
            "ed_volume_cm3": "Peritumoral Edema Volume (ED, cm³)",
            "ncr_volume_cm3": "Necrotic Core Volume (NCR, cm³)",
            "et_to_wt_ratio": "Enhancing Fraction (ET / WT)",
            "tc_to_wt_ratio": "Core Fraction (TC / WT)",
            "ed_to_wt_ratio": "Edema Fraction (ED / WT)",
        }

        for metric, display_name in metric_display_names.items():
            if metric not in df.columns:
                continue

            tot_vals = df[metric].dropna()
            tot_med = round(float(tot_vals.median()), 2)
            tot_iqr = self._compute_iqr(tot_vals)
            tot_mean = round(float(tot_vals.mean()), 2)
            tot_std = round(float(tot_vals.std()), 2)

            comp = comparison.get(metric, {})
            hgg_med = comp.get("hgg_median", 0.0)
            hgg_iqr = comp.get("hgg_iqr", 0.0)
            lgg_med = comp.get("lgg_median", 0.0)
            lgg_iqr = comp.get("lgg_iqr", 0.0)
            p_val = comp.get("p_value", 1.0)

            p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
            sig_str = "*** (p<0.001)" if p_val < 0.001 else ("* (p<0.05)" if p_val < 0.05 else "ns")

            rows.append({
                "Parameter": f"{display_name} [Median (IQR)]",
                "Cohort_Total": f"{tot_med} ({tot_iqr})",
                "HGG": f"{hgg_med} ({hgg_iqr})",
                "LGG": f"{lgg_med} ({lgg_iqr})",
                "P_Value": p_str,
                "Significance": sig_str,
            })

            rows.append({
                "Parameter": f"  └─ Mean ± Std",
                "Cohort_Total": f"{tot_mean} ± {tot_std}",
                "HGG": f"{comp.get('hgg_mean', 0.0)} ± {comp.get('hgg_std', 0.0)}",
                "LGG": f"{comp.get('lgg_mean', 0.0)} ± {comp.get('lgg_std', 0.0)}",
                "P_Value": "-",
                "Significance": "-",
            })

        # Demographics: Age
        if "age" in df.columns:
            age_valid = df["age"].dropna()
            hgg_age = hgg_df["age"].dropna()
            lgg_age = lgg_df["age"].dropna()

            if len(hgg_age) > 0 and len(lgg_age) > 0:
                _, p_age = stats.mannwhitneyu(hgg_age, lgg_age, alternative="two-sided")
                p_age_str = "< 0.001" if p_age < 0.001 else f"{p_age:.4f}"
                sig_age = "***" if p_age < 0.001 else ("*" if p_age < 0.05 else "ns")
            else:
                p_age_str, sig_age = "N/A", "N/A"

            rows.append({
                "Parameter": "Patient Age (years) [Mean ± Std]",
                "Cohort_Total": f"{age_valid.mean():.1f} ± {age_valid.std():.1f}" if len(age_valid) else "N/A",
                "HGG": f"{hgg_age.mean():.1f} ± {hgg_age.std():.1f}" if len(hgg_age) else "N/A",
                "LGG": f"{lgg_age.mean():.1f} ± {lgg_age.std():.1f}" if len(lgg_age) else "N/A",
                "P_Value": p_age_str,
                "Significance": sig_age,
            })

        return pd.DataFrame(rows)
