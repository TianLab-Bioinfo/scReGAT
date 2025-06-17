# Overview

All relevant textual and network data involved in regulatory network construction and downstream analysis are provided in this directory.

## Data Sources

### cRE–Gene Interaction Datasets

| File | Description |
|------|-------------|
| `PO.txt` | Promoter–Other regulatory element (cRE) links integrated from Hi-C data; used to map distal cREs to gene promoters. |
| `PP.txt` | Promoter–Promoter interactions inferred from Hi-C data; used to capture co-regulatory promoter coordination. |
| `all_tissue_SNP_Gene.txt.gz` | GTEx-derived SNP–gene linkage data across all tissues; used for integrating genetic variants with gene regulation. |

### TF–Gene Regulation Datasets

| File | Description |
|------|-------------|
| `trrust_rawdata.human.tsv` | Transcription factor–target gene interactions curated from the TRRUST database (human only). |
| `TF_Gene_tissue_cutoff1.csv` | TF–gene regulatory associations derived from ChEA3 ChIP-seq datasets across multiple tissues; filtered using cutoff = 1. |

### Core Reference Files

| File | Description |
|------|-------------|
| `hg38.chrom.sizes` | Chromosome size reference file for the hg38 human genome build; required for genomic range operations. |
| `genes.protein.tss.tsv` | Table of protein-coding genes with annotated transcription start sites (TSS); used for promoter and gene mapping. |

### Disease-Associated Cohort File

| File | Description |
|------|-------------|
| `Cohort_survival.tat.bz2` | Patient survival cohort for neuroblastoma (case group); used for survival analysis and case-based prioritization. |

## GWAS Datasets (`/GWAS/` directory)

The `GWAS` subfolder includes disease-associated SNP datasets downloaded from public genome-wide association studies:

| Subfolder | Description |
|-----------|-------------|
| `SCZ/` | Schizophrenia (SCZ) risk loci from GWAS summary statistics. |
| `AD/` | Alzheimer's disease (AD) GWAS SNP data. |
| `MS/` | Multiple sclerosis (MS) GWAS-associated variant information. |

## Cell Type–Specific cRE–Gene Interactions (`/celltype_specific_cRE_interactions/` directory)

This folder contains cRE–gene interaction maps validated by PLAC-seq experiments across brain cell types:

| Subfolder | Description |
|-----------|-------------|
| `microglia/` | Experimentally supported enhancer–promoter links specific to microglia. |
| `neurons/` | PLAC-seq-based cRE–gene interactions in neurons. |
| `oligodendrocytes/` | Oligodendrocyte-specific cRE–gene regulatory links. |
