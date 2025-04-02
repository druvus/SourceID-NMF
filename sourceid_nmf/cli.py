"""
Command-line interface for SourceID-NMF with advanced optimization options.

This module provides the command-line interface for running SourceID-NMF.
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional

from sourceid_nmf import __version__
from sourceid_nmf.core import run_source_tracking
from sourceid_nmf.estimation import evaluate_performance, save_performance_results
from sourceid_nmf.utils import configure_logging

logger = logging.getLogger("sourceid_nmf.cli")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="""SourceID-NMF: Microbial source tracking via non-negative matrix factorization.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Track command - main functionality
    track_parser = subparsers.add_parser(
        "track",
        help="Perform source tracking analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    track_parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input count table (tab-separated)"
    )

    track_parser.add_argument(
        "-n", "--name",
        required=True,
        help="Path to sample name file (tab-separated)"
    )

    track_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to output the estimated proportions"
    )

    track_parser.add_argument(
        "-m", "--mode",
        choices=["normal", "cluster"],
        default="normal",
        help="Whether to optimize using clustering algorithms"
    )

    track_parser.add_argument(
        "-f", "--cutoff",
        type=float,
        default=0.25,
        help="Threshold for clustering algorithms"
    )

    track_parser.add_argument(
        "-t", "--thread",
        type=int,
        default=20,
        help="Number of threads for multiprocessing"
    )

    track_parser.add_argument(
        "-e", "--iter",
        type=int,
        default=2000,
        help="Maximum number of iterations for the NMF model"
    )

    track_parser.add_argument(
        "-r", "--rho",
        type=float,
        default=1.0,
        help="Initial penalty parameter (0 for auto-selection)"
    )

    track_parser.add_argument(
        "-a", "--weight",
        type=int,
        default=1,
        help="Weighting matrix factor"
    )

    track_parser.add_argument(
        "-c", "--threshold",
        type=float,
        default=1e-06,
        help="Convergence threshold for Lagrangian functions"
    )

    track_parser.add_argument(
        "-p", "--perf",
        help="Path to output performance metrics (optional)"
    )

    # Advanced optimization options
    optimization_group = track_parser.add_argument_group('Advanced Optimization Options')

    optimization_group.add_argument(
        "--use-active-set",
        action="store_true",
        help="Enable active-set method to accelerate sparse data processing"
    )

    optimization_group.add_argument(
        "--no-active-set",
        action="store_true",
        help="Disable active-set method (overrides auto-detection)"
    )

    optimization_group.add_argument(
        "--adaptive-rho",
        action="store_true",
        help="Enable adaptive rho parameter for faster convergence"
    )

    optimization_group.add_argument(
        "--fixed-rho",
        action="store_true",
        help="Use fixed rho parameter (no adaptation)"
    )

    optimization_group.add_argument(
        "--auto-optimize",
        action="store_true",
        help="Automatically detect and apply optimal settings (default)"
    )

    # Evaluate command - for evaluation against true proportions
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate tracking performance against true proportions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    evaluate_parser.add_argument(
        "-e", "--estimated",
        required=True,
        help="Path to estimated proportions file"
    )

    evaluate_parser.add_argument(
        "-t", "--true",
        required=True,
        help="Path to true proportions file"
    )

    evaluate_parser.add_argument(
        "-o", "--output",
        default="proportion_perf.txt",
        help="Path to output performance results"
    )

    # Global arguments
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times)"
    )

    parser.add_argument(
        "--log",
        help="Path to log file (optional)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"SourceID-NMF {__version__}"
    )

    # Parse arguments
    args = parser.parse_args()

    # Require a command
    if not args.command:
        parser.print_help()
        sys.exit(1)

    return args


def main() -> None:
    """
    Main entry point for the command-line interface.
    """
    # Parse arguments
    args = parse_args()

    # Configure logging
    configure_logging(args.verbose, args.log)

    try:
        # Execute the appropriate command
        if args.command == "track":
            logger.info("Starting source tracking")

            # Determine optimization settings
            use_active_set = not args.no_active_set
            adaptive_rho = not args.fixed_rho

            # Explicit settings override defaults
            if args.use_active_set:
                use_active_set = True
            if args.adaptive_rho:
                adaptive_rho = True

            # Use auto-detect if requested
            if args.auto_optimize:
                use_active_set = True
                adaptive_rho = True
                if args.rho == 1.0:  # If using default rho
                    args.rho = 0  # 0 means auto-selection

            # Log optimization settings
            logger.info(f"Using optimization settings: active-set={use_active_set}, "
                       f"adaptive-rho={adaptive_rho}, initial-rho={args.rho}")

            run_source_tracking(
                data_path=args.input,
                name_path=args.name,
                output_path=args.output,
                mode=args.mode,
                cutoff=args.cutoff,
                thread=args.thread,
                iteration=args.iter,
                rho=args.rho,
                weight_factor=args.weight,
                threshold=args.threshold,
                perf_output=args.perf,
                use_active_set=use_active_set,
                adaptive_rho=adaptive_rho
            )
            logger.info(f"Source tracking completed. Results saved to {args.output}")

        elif args.command == "evaluate":
            logger.info("Starting performance evaluation")
            jsd_avg, jsd, diff_avg, diff = evaluate_performance(
                estimated_proportions_path=args.estimated,
                true_proportions_path=args.true
            )

            # Print summary to console
            print(f"Average JSD between estimated and true proportions: {jsd_avg:.6f}")
            print(f"Average difference between estimated and true proportions: {diff_avg:.6f}")

            # Extract row labels from the estimated proportions file
            import pandas as pd
            estimated_props = pd.read_csv(args.estimated, sep=" ", header=0, index_col=0)
            row_labels = list(estimated_props.index.values)

            # Save detailed results
            save_performance_results(jsd, diff, row_labels, args.output)
            logger.info(f"Evaluation completed. Results saved to {args.output}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()