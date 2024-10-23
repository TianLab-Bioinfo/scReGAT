import pickle
import os
import pandas as pd
import torch
import anndata as ad
from torch_geometric.loader import DataLoader
import scanpy as sc
from scregat.data_process import prepare_model_input,sum_counts,plot_edge, ATACGraphDataset
import numpy as np
from scregat.model import train_scregat, explain_model_ig



with open('./dataset_atac.pkl', 'rb') as f:
    dataset_atac = pickle.load(f)

gene_peaks = [peak for peak in dataset_atac.array_peak if not peak.startswith('chr')]
chr_peaks = [peak for peak in dataset_atac.array_peak if peak.startswith('chr')]
gene_df = pd.read_csv('../pre_data/genes.protein.tss.tsv', sep='\t', header=None)
positions = [i for i, val in enumerate(gene_df[2].values) if val in gene_peaks]
chromatins = gene_df[0].iloc[positions]
centers = gene_df[1].iloc[positions]

 def calculate_center_and_adjust_for_promoter(chromatins, centers, shift=512):
        result = []
        for chrom, center in zip(chromatins, centers):
            start_adjusted = center - shift
            end_adjusted = center + shift
            result.append((chrom, start_adjusted, end_adjusted))
        return result

def calculate_center_and_adjust_for_enhancer(peaks, shift=512):
        result = []
        for peak in peaks:
            chrom, start, end = peak.split('-')
            start = int(start)
            end = int(end)
            center = (start + end) // 2
            start_adjusted = center - shift
            end_adjusted = center + shift
            result.append((chrom, start_adjusted, end_adjusted))
        return result    

adjusted_promoter = calculate_center_and_adjust_for_promoter(chromatins, centers)
adjusted_peaks = calculate_center_and_adjust_for_enhancer(chr_peaks)
combined_data = adjusted_promoter + adjusted_peaks
with open('center.bed', 'w') as f:
    for chrom, start, end in combined_data:
        f.write(f"{chrom}\t{start}\t{end}\n")   

# next, user should run bedtools in your device to get .fa.
bedtools getfasta -fi hg38.fa -bed ~/scReGAT/10x/center.bed -fo ~/scReGAT/10x/center.fa
