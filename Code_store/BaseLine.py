import os
import pandas as pd
import torch
import anndata as ad
from torch_geometric.loader import DataLoader
import scanpy as sc
from scregat.data_process import prepare_model_input,sum_counts,plot_edge, ATACGraphDataset
import numpy as np
import pickle

with open('./dataset_atac.pkl', 'rb') as f:
    dataset_atac = pickle.load(f)

dataset_graph = ATACGraphDataset('./input_graph/')

gene_peaks = [peak for peak in dataset_atac.array_peak if not peak.startswith('chr')]
chr_peaks = [peak for peak in dataset_atac.array_peak if peak.startswith('chr')]

# user 输入genes.protein.tss.tsv的位置
gene_df = pd.read_csv('../pre_data/genes.protein.tss.tsv', sep='\t', header=None)

positions = [i for i, val in enumerate(gene_df[2].values) if val in gene_peaks]
chromatins = gene_df[0].iloc[positions]
centers = gene_df[1].iloc[positions]

# user 输入genes.protein.tss.tsv的位置
gene_df = pd.read_csv('../pre_data/genes.protein.tss.tsv', sep='\t', header=None)

gene_peaks = [peak for peak in dataset_atac.array_peak if not peak.startswith('chr')]  
chr_peaks = [peak for peak in dataset_atac.array_peak if peak.startswith('chr')]
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

# 获取peak序列信息
with open('center.bed', 'w') as f:
    for chrom, start, end in combined_data:
        f.write(f"{chrom}\t{start}\t{end}\n")   

# 在命令行外运行
bedtools getfasta -fi hg38.fa -bed ~/scReGAT/10x/center.bed -fo ~/scReGAT/10x/center.fa
# ~/scReGAT/10x/center.bed 是我们刚刚生成的文件
# ~/scReGAT/10x/center.fa 生成的序列文件

# 现在我们需要生成DNABERT2向量
# 打开 https://huggingface.co/zhihan1996/DNABERT-2-117M/tree/main


seq_vec = torch.load('./seq.pth')
