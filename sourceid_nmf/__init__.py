"""
SourceID-NMF: Towards more accurate microbial source tracking via non-negative matrix factorization.

A tool for precise microbial source tracking that utilizes a non-negative matrix factorization (NMF)
algorithm to trace the microbial sources contributing to a target sample.
"""

__version__ = "0.1.0"

from sourceid_nmf.core import run_source_tracking
from sourceid_nmf.estimation import evaluate_performance

__all__ = ["run_source_tracking", "evaluate_performance"]