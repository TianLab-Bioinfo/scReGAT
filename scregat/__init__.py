"""
scReGAT: Single-cell Regulatory Graph Attention Network

A package for analyzing single-cell ATAC-seq data and predicting gene expression
using graph attention networks.
"""

__version__ = "0.1.0"

# Import main classes and functions
from .model import SCReGAT, EdgeDiversityLoss1, EdgeDiversityLoss2, set_seed
from .data_process import (
    ATACDataset,
    ATACGraphDataset,
    prepare_model_input,
    sum_counts,
    plot_edge,
    tfidf,
    lsi
)
from .train import (
    train_model,
    train_model_sample,
    test_model,
    test_model_for_gene_loss,
    get_gene_expression,
    get_edge_info,
    get_edge_info_all,
    get_edge_info_with_anchor,
    process_samples,
    preprocess_graph,
    process_data,
    get_peaks_df,
    compute_summed_peaks,
    generate_peaks_list,
    expand_features,
    process_genes_and_select_rows,
    map_celltypes_to_unique_numbers,
    create_ref_graph,
    Create_ref_graph,
    get_overlap_peaks,
    save_or_load_weights,
    process_edge_names
)

__all__ = [
    # Model
    "SCReGAT",
    "EdgeDiversityLoss1",
    "EdgeDiversityLoss2",
    "set_seed",
    # Data processing
    "ATACDataset",
    "ATACGraphDataset",
    "prepare_model_input",
    "sum_counts",
    "plot_edge",
    "tfidf",
    "lsi",
    # Training and testing
    "train_model",
    "train_model_sample",
    "test_model",
    "test_model_for_gene_loss",
    "get_gene_expression",
    # Utility functions
    "get_edge_info",
    "get_edge_info_all",
    "get_edge_info_with_anchor",
    "process_samples",
    "preprocess_graph",
    "process_data",
    "get_peaks_df",
    "compute_summed_peaks",
    "generate_peaks_list",
    "expand_features",
    "process_genes_and_select_rows",
    "map_celltypes_to_unique_numbers",
    "create_ref_graph",
    "Create_ref_graph",
    "get_overlap_peaks",
    "save_or_load_weights",
    "process_edge_names",
]

