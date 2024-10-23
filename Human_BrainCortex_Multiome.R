library(Seurat)
library(SeuratDisk)

# Dataset source:
# scATAC-seq GSE204682
# scRNA-seq https://cellxgene.cziscience.com/collections/ceb895f4-ff9f-403a-b7c3-187a9657ac2c
# 10x scRNA-seq from human cortex
# eQTL: GTex GTEx_Analysis_v8_eQTL

mt <- readRDS('./GSE204682_count_matrix.RDS')
atac <- CreateSeuratObject(mt)

rna <- readRDS('./484dbc33-c7dc-4e5e-9954-7f2a1cc849bc.rds')
rna$author_cell_type

eqtl <- read.table('./Brain_Cortex.v8.signif_variant_gene_pairs.txt', header = T)

eqtl$gene_id_clean <- sapply(strsplit(as.character(eqtl$gene_id), "\\."), `[`, 1)

write.csv(rownames(rna)[rownames(rna) %in% unique(eqtl$gene_id_clean)], file = 'gene_select.csv')

select_genes <- rownames(rna)[rownames(rna) %in% unique(eqtl$gene_id_clean)]

atac@meta.data$celltype <- as.vector(rna$author_cell_type)
atac@meta.data$celltype_rna <- as.vector(rna$author_cell_type)
rna@meta.data$celltype <- as.vector(rna$author_cell_type)
SaveH5Seurat(atac, filename = 'ATAC.h5Seurat',overwrite = T)
SaveH5Seurat(rna, filename = 'RNA.h5Seurat',overwrite = T)
Convert('RNA.h5Seurat', dest = "h5ad",overwrite = T)
Convert('ATAC.h5Seurat', dest = "h5ad",overwrite = T)
