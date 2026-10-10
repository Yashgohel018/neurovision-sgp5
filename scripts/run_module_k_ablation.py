"""
NeuroVision AI: Module K Runner - Ablation Study
Executes systematic ablation experiments and produces ablation_results.csv and summary chart.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import setup_logger
from src.module_k_ablation.ablation import run_ablation_benchmarks, plot_ablation_summary

logger = setup_logger("RunModuleK")


def main():
    logger.info("=" * 80)
    logger.info("   NeuroVision AI: MultiModal-BrainNet Pipeline")
    logger.info("   Module K: Systematic Ablation Study")
    logger.info("=" * 80)

    df = run_ablation_benchmarks()
    csv_out = os.path.join(PROJECT_ROOT, "reports", "ablation_results.csv")
    df.to_csv(csv_out, index=False)
    logger.info(f"Saved ablation results to: {csv_out}")

    fig_out = os.path.join(PROJECT_ROOT, "reports", "figures", "ablation_comparison.png")
    plot_ablation_summary(df, fig_out)
    logger.info(f"Saved ablation comparison figure to: {fig_out}")

    logger.info("Module K completed successfully.")


if __name__ == "__main__":
    main()
