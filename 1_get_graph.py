import os
import pandas as pd
import torch
import anndata as ad
from torch_geometric.loader import DataLoader
import scanpy as sc
from scregat.data_process import prepare_model_input,sum_counts,plot_edge, ATACGraphDataset
import numpy as np

atac_file = './Pancreas_ATAC.h5ad'
RNA_h5ad_file = "./Pancreas_RNA.h5ad"
adata_rna = sc.read_h5ad(RNA_h5ad_file)
adata_rna.obs['celltype'] = adata_rna.obs['celltype'].astype('object')
df_rna = sum_counts(adata_rna,by = 'celltype',marker_gene_num=300)

# Please ensure that your gene characters are gene symbols and not gene IDs
# or you can transfer gene id to gene symbols by gene annotation file


dataset_atac, dataset_graph = prepare_model_input(
    path_data_root = './' ,
    file_atac = atac_file, 
    df_rna_celltype = df_rna,
    path_eqtl = '../all_tissue_SNP_Gene.txt',
    hg19tohg38 = False, min_percent = 0.01)


