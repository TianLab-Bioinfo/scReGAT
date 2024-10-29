# find marker cres
cre_obj <- CreateSeuratObject(pair_cell_normalize, meta.data = data.frame(group = scregat_ATAC[,colnames(pair_cell_normalize)]$celltype, row.names = colnames(pair_cell_normalize)))
Idents(cre_obj) <- cre_obj$group
TtoN_cre_markers <- FindMarkers(cre_obj, ident.1 = 'T_Epithelial', ident.2 = 'N_Epithelial',min.pct = 0, logfc.threshold = 0)
TtoP_cre_markers <- FindMarkers(cre_obj, ident.1 = 'T_Epithelial', ident.2 = 'P_Epithelial',min.pct = 0, logfc.threshold = 0)
PtoN_cre_markers <- FindMarkers(cre_obj, ident.1 = 'P_Epithelial', ident.2 = 'N_Epithelial',min.pct = 0, logfc.threshold = 0)

avg_log2FC = 0.1
T_up_cre <- TtoN_cre_markers[TtoN_cre_markers$avg_log2FC > avg_log2FC & TtoN_cre_markers$p_val_adj < 0.05,]
T_down_cre <- TtoN_cre_markers[TtoN_cre_markers$avg_log2FC < -avg_log2FC & TtoN_cre_markers$p_val_adj < 0.05,]
P_up_cre <- PtoN_cre_markers[PtoN_cre_markers$avg_log2FC > avg_log2FC & PtoN_cre_markers$p_val_adj < 0.05,]
P_down_cre <- PtoN_cre_markers[PtoN_cre_markers$avg_log2FC < -avg_log2FC & PtoN_cre_markers$p_val_adj < 0.05,]

T_up_genes <- str_sub(rownames(T_up_cre),3,-3) %>% str_split("','") %>% sapply(function(x){x[1]})
T_up_peaks <- str_sub(rownames(T_up_cre),3,-3) %>% str_split("','") %>% sapply(function(x){x[2]})
T_down_genes <- str_sub(rownames(T_down_cre),3,-3) %>% str_split("','") %>% sapply(function(x){x[1]})
T_down_peaks <- str_sub(rownames(T_down_cre),3,-3) %>% str_split("','") %>% sapply(function(x){x[2]})
P_up_genes <- str_sub(rownames(P_up_cre),3,-3) %>% str_split("','") %>% sapply(function(x){x[1]})
P_up_peaks <- str_sub(rownames(P_up_cre),3,-3) %>% str_split("','") %>% sapply(function(x){x[2]})
P_down_genes <- str_sub(rownames(P_down_cre),3,-3) %>% str_split("','") %>% sapply(function(x){x[1]})
P_down_peaks <- str_sub(rownames(P_down_cre),3,-3) %>% str_split("','") %>% sapply(function(x){x[2]})


# load clinical info and SNV
load(file = './output/TCGA_data_out/lqm_all_variants.Rdata') # all_variants
survival = as.data.frame(fread('../data/bulk/TCGA/TCGA/survival%2FCOADREAD_survival.txt'))
rownames(survival) = survival$sample
clinical = as.data.frame(fread('../data/bulk/TCGA/TCGA/TCGA.COAD.sampleMap%2FCOAD_clinicalMatrix'))
rownames(clinical) = clinical$sampleID


SNV.bed = unique(all_variants[, c('CHROM','POS')])
SNV.bed$POS = as.integer(SNV.bed$POS)
SNV.bed$Start = SNV.bed$POS - 1
SNV.bed$End = SNV.bed$POS + 1 
SNV.bed$POS = NULL
SNV.bed = bt.sort(SNV.bed)
SNV.bed$id = 1:nrow(SNV.bed)

diff_peak.bed = strsplit2(c(T_up_peaks), '-')
diff_peak.bed = as.data.frame(diff_peak.bed)


# peak上突变越多是不是生存越差
# result1 = bt.intersect(a = bt.sort(SNV.bed),b = bt.sort(diff_peak.bed), wa = TRUE, wb = TRUE)
# snv_freq <- as.data.frame(table(all_variants[all_variants$SNV %in% paste0(SNV.bed[result1$V4,'V1'], ':', SNV.bed[result1$V4,'V2']+1),'names']))
dist <- bt.closest(a = bt.sort(SNV.bed), b = bt.sort(diff_peak.bed), D = "ref")
dist <- dist[dist$V6 != -1,]
snv_freq <- as.data.frame(table(all_variants[all_variants$SNV %in% paste0(SNV.bed[dist[abs(dist$V8) < 1000,'V4'],'V1'], ':', SNV.bed[dist[abs(dist$V8) < 1000,'V4'],'V2']+1),'names']))
rownames(snv_freq) = snv_freq$Var1

tmp = as.data.frame(table(all_variants$names))
rownames(tmp) = tmp$Var1
snv_freq$Freq <- snv_freq$Freq / tmp[snv_freq$Var1,'Freq']

options(repr.plot.width=5, repr.plot.height=6, repr.plot.res=300)
CRC_subtype <- data.frame(group=rep('high',length(rownames(snv_freq))), Dead=survival[rownames(snv_freq),]$OS, OS=survival[rownames(snv_freq),]$OS.time)
CRC_subtype$group[snv_freq$Freq < median(snv_freq$Freq)] <- 'low'
CRC_subtype$Dead <- as.numeric(CRC_subtype$Dead)
CRC_subtype$OS <- as.numeric(CRC_subtype$OS)
CRC_subtype <- CRC_subtype[!is.na(CRC_subtype$Dead),]

survobj <- with(CRC_subtype, Surv(OS,Dead))
fit <- survfit(survobj~group, data=CRC_subtype)
p_values <- c()
p_values <- c(p_values, surv_pvalue(fit)$pval)
# if (surv_pvalue(fit)$pval < 0.05){
    p.sub <- ggsurvplot(fit, data = CRC_subtype, pval = TRUE, pval.method = TRUE, risk.table = TRUE)+labs(title = col)
    p.sub1 <- wrap_elements(p.sub$plot+p.sub$table+plot_layout(ncol = 1, heights = c(5,1))&labs(y="",title=''))
    print(p.sub1)
# }


SNV.bed = unique(all_variants[all_variants$names %in% intersect(clinical[clinical$CDE_ID_3226963 %in% c('MSS', 'MSI-L'), 'sampleID'], all_variants$names), c('CHROM','POS')])
SNV.bed$POS = as.integer(SNV.bed$POS)
SNV.bed$Start = SNV.bed$POS - 1
SNV.bed$End = SNV.bed$POS + 1 
SNV.bed$POS = NULL
SNV.bed = bt.sort(SNV.bed)
SNV.bed$id = 1:nrow(SNV.bed)


# result1 = bt.intersect(a = bt.sort(SNV.bed),b = bt.sort(diff_peak.bed), wa = TRUE, wb = TRUE)
# snv_freq <- as.data.frame(table(all_variants[all_variants$SNV %in% paste0(SNV.bed[result1$V4,'V1'], ':', SNV.bed[result1$V4,'V2']+1),'names']))
dist <- bt.closest(a = bt.sort(SNV.bed), b = bt.sort(diff_peak.bed), D = "ref")
dist <- dist[dist$V6 != -1,]
snv_freq <- as.data.frame(table(all_variants[all_variants$SNV %in% paste0(SNV.bed[dist[abs(dist$V8) < 1000,'V4'],'V1'], ':', SNV.bed[dist[abs(dist$V8) < 1000,'V4'],'V2']+1),'names']))
rownames(snv_freq) = snv_freq$Var1

tmp = as.data.frame(table(all_variants$names))
rownames(tmp) = tmp$Var1
snv_freq$Freq <- snv_freq$Freq / tmp[snv_freq$Var1,'Freq']

options(repr.plot.width=5, repr.plot.height=6, repr.plot.res=300)
CRC_subtype <- data.frame(group=rep('high',length(rownames(snv_freq))), Dead=survival[rownames(snv_freq),]$OS, OS=survival[rownames(snv_freq),]$OS.time)
CRC_subtype$group[snv_freq$Freq < median(snv_freq$Freq)] <- 'low'
CRC_subtype$Dead <- as.numeric(CRC_subtype$Dead)
CRC_subtype$OS <- as.numeric(CRC_subtype$OS)
CRC_subtype <- CRC_subtype[!is.na(CRC_subtype$Dead),]

survobj <- with(CRC_subtype, Surv(OS,Dead))
fit <- survfit(survobj~group, data=CRC_subtype)
p_values <- c(p_values, surv_pvalue(fit)$pval)
# if (surv_pvalue(fit)$pval < 0.05){
    p.sub <- ggsurvplot(fit, data = CRC_subtype, pval = TRUE, pval.method = TRUE, risk.table = TRUE)+labs(title = col)
    p.sub1 <- wrap_elements(p.sub$plot+p.sub$table+plot_layout(ncol = 1, heights = c(5,1))&labs(y="",title=''))
    print(p.sub1)
# }