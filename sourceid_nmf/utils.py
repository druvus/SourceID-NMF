"""
Utility functions for the SourceID-NMF package.
"""

import logging
import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Create a logger
logger = logging.getLogger("sourceid_nmf")


def configure_logging(verbosity: int = 0, log_file: Optional[str] = None) -> None:
    """
    Configure logging based on verbosity level.

    Args:
        verbosity: Verbosity level (0=WARNING, 1=INFO, 2=DEBUG)
        log_file: Optional path to log file
    """
    # Convert verbosity to logging levels (0=WARNING, 1=INFO, 2=DEBUG)
    log_level = max(2 - verbosity, 0) * 10 + 10  # 30=WARNING, 20=INFO, 10=DEBUG

    # Create formatter
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Always add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logger.debug(f"Logging configured with level {logging.getLevelName(log_level)}")


def load_data(data_path: str, name_path: str) -> tuple:
    """
    Load and parse input data files.

    Args:
        data_path: Path to the count table file
        name_path: Path to the sample name file

    Returns:
        tuple: (sources, sinks, sources_label, sinks_label)
    """
    logger.info(f"Loading data from {data_path} and {name_path}")

    try:
        data = pd.read_csv(data_path, sep="\t", header=0, index_col=0)
        name = pd.read_csv(name_path, sep="\t", header=0)
    except Exception as e:
        logger.error(f"Error loading input files: {e}")
        raise

    name_id = list(name.loc[:, 'SampleID'])
    source_sink_id = list(name.loc[:, 'SourceSink'])

    sources = []
    sinks = []
    sources_label = []
    sinks_label = []

    for i in range(len(source_sink_id)):
        if source_sink_id[i] == 'Source':
            source = data[name_id[i]].values
            sources.append(source)
            sources_label.append(name_id[i])
        elif source_sink_id[i] == 'Sink':
            sink = data[name_id[i]].values
            sinks.append(sink)
            sinks_label.append(name_id[i])
        else:
            logger.warning(f"Skipping unknown category: {source_sink_id[i]}")

    # Organize and summarize all sources to prepare inputs for the model
    sources = np.array(sources).T
    # Organize and summarize all sinks to prepare inputs for the model
    sinks = np.array(sinks).T

    # Add unknown sources to the source's label
    sources_label.append('unknown')

    logger.info(f"Loaded {sources.shape[1]} sources and {sinks.shape[1]} sinks")
    return sources, sinks, sources_label, sinks_label


def save_results(data: np.ndarray, index: List[str], columns: List[str], output_path: str) -> None:
    """
    Save results to a file.

    Args:
        data: Data to save
        index: Row index labels
        columns: Column labels
        output_path: Path to save the file
    """
    logger.info(f"Saving results to {output_path}")
    try:
        result_df = pd.DataFrame(data, index=index, columns=columns)
        result_df.to_csv(output_path, sep=' ')
        logger.debug(f"Results saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise