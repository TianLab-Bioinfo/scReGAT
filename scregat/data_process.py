import os
import random
import itertools
from collections import defaultdict
from typing import Optional, Mapping, List, Union

import numpy as np
import pandas as pd
from pandas import DataFrame

import torch
from torch_geometric.data import InMemoryDataset, Data

import scanpy as sc
import episcanpy.api as epi
import anndata as ad
from anndata import AnnData

from scipy import sparse
import sklearn
from statsmodels.distributions.empirical_distribution import ECDF

import cosg


def sum_counts(adata, by="celltype", use_marker_genes=True, marker_gene_num=300):
    if use_marker_genes:
        # run cosg
        cosg.cosg(
            adata,
            key_added="cosg",
            mu=1,
            remove_lowly_expressed=True,
            expressed_pct=0.2,
            n_genes_user=marker_gene_num,
            use_raw=True,
            groupby=by,
        )
        # keep cosg genes
        cosg_genes = []
        for i in range(adata.uns["cosg"]["names"].shape[0]):
            cosg_genes.extend(list(adata.uns["cosg"]["names"][i]))
        cosg_genes = list(set(cosg_genes))
        adata = adata[:, cosg_genes]
    # sum counts
    df_rna = pd.DataFrame(
        adata.X.toarray(), index=adata.obs.index, columns=adata.var.index
    )
    df_rna_cell = df_rna
    df_rna_cell["celltype"] = adata.obs.loc[:, by]
    df_rna_cell = df_rna_cell.groupby(by).apply(lambda x: x.sum())
    df_rna_cell
    df_rna_celltype = df_rna_cell.iloc[:, :-1]
    return df_rna_celltype


def plot_edge(df_weight, dataset_atac):
    df_weight_in = df_weight.copy()
    adata_edge = ad.AnnData(
        X=df_weight_in, obs=dataset_atac.adata.obs.loc[df_weight_in.index, :]
    )
    adata = adata_edge
    sc.pp.highly_variable_genes(adata, n_top_genes=10000, flavor="seurat")
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack", n_comps=20)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20, metric="cosine")
    sc.tl.umap(adata, min_dist=0.5)
    sc.pl.umap(adata, color=["nb_features", "celltype"])
    return


def tfidf(X: Union[np.ndarray, sparse.spmatrix]) -> Union[np.ndarray, sparse.spmatrix]:
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    Parameters
    ----------
    X
        Input matrix
    Returns
    -------
    X_tfidf
        TF-IDF normalized matrix
    """
    idf = X.shape[0] / X.sum(axis=0)
    if sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def lsi(
    adata: AnnData,
    n_components: int = 20,
    use_top_features: Optional[bool] = False,
    min_cutoff: float = 0.05,
    **kwargs,
) -> None:
    r"""
    LSI analysis
    Parameters
    ----------
    adata
        Input dataset
    n_components
        Number of dimensions to use
    use_top_features
        Whether to find most frequently observed features and use them
    min_cutoff
        Cutoff for feature to be included in the ``adata.var['select_feature']``.
        For example, '0.05' to set the top 95% most common features as the selected features.
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.utils.extmath.randomized_svd`
    """
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior

    adata_use = adata.copy()
    if use_top_features:
        adata_use.var["featurecounts"] = np.array(np.sum(adata_use.X, axis=0))[0]
        df_var = adata_use.var.sort_values(by="featurecounts")
        ecdf = ECDF(df_var["featurecounts"])
        df_var["percentile"] = ecdf(df_var["featurecounts"])
        df_var["selected_feature"] = df_var["percentile"] > min_cutoff
        adata_use.var = df_var.loc[adata_use.var.index, :]

    # factor_size = int(np.median(np.array(np.sum(adata_use.X, axis=1))))
    X_norm = np.log1p(tfidf(adata_use.X) * 1e4)
    if use_top_features:
        X_norm = X_norm.toarray()[:, adata_use.var["selected_feature"]]
    else:
        X_norm = X_norm.toarray()
    svd = sklearn.decomposition.TruncatedSVD(
        n_components=n_components, algorithm="arpack"
    )
    X_lsi = svd.fit_transform(X_norm)
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi


class ATACDataset(object):
    def __init__(self, adata_atac, raw_filename: str, data_root: str, file_chrom: str):
        self.data_root = data_root
        self.raw_filename = raw_filename
        self.adata = adata_atac
        # self.adata.raw = self.adata.copy()
        self.path_process = os.path.join(data_root, "processed_files")
        if not os.path.exists(self.path_process):
            os.mkdir(self.path_process)
        self.file_peaks_sort = os.path.join(self.path_process, "peaks.sort.bed")
        if os.path.exists(self.file_peaks_sort):
            os.remove(self.file_peaks_sort)
        self.file_chrom = file_chrom
        # tools - these should be provided by the user or set via environment variables
        # Default to None, user should set these paths if needed
        self.bedtools = os.environ.get(
            "BEDTOOLS_PATH", "bedtools"
        )  # Default to system bedtools
        self.liftover = os.environ.get("LIFTOVER_PATH", None)
        self.file_chain = os.environ.get("LIFTOVER_CHAIN", None)
        self.generate_peaks_file()
        self.all_promoter_genes = None
        self.all_proximal_genes = None
        self.adata_merge = None
        self.other_peaks = None
        self.df_graph = None
        self.list_graph = None
        self.array_peak = None
        self.array_celltype = None
        self.df_rna = None
        self.dict_promoter = None
        self.df_gene_peaks = None
        self.df_proximal = None
        self.df_distal = None
        self.df_eqtl = None
        self.df_tf = None

    def generate_peaks_file(self):
        df_chrom = pd.read_csv(self.file_chrom, sep="\t", header=None, index_col=0)
        df_chrom = df_chrom.iloc[:24]
        file_peaks_atac = os.path.join(self.path_process, "peaks.bed")
        fmt_peak = "{chrom_peak}\t{start_peak}\t{end_peak}\t{peak_id}\n"
        with open(file_peaks_atac, "w") as w_peak:
            for one_peak in self.adata.var.index:
                chrom_peak = one_peak.strip().split("-")[0]
                # locs = one_peak.strip().split(':')[1]
                if chrom_peak in df_chrom.index:
                    start_peak = one_peak.strip().split("-")[1]
                    end_peak = one_peak.strip().split("-")[2]
                    peak_id = one_peak
                    w_peak.write(fmt_peak.format(**locals()))

        os.system(f"{self.bedtools} sort -i {file_peaks_atac} > {self.file_peaks_sort}")

    def hg19tohg38(self):
        path_peak = os.path.join(self.data_root, "peaks_process")
        if not os.path.exists(path_peak):
            os.mkdir(path_peak)

        file_ummap = os.path.join(path_peak, "unmap.bed")
        file_peaks_hg38 = os.path.join(path_peak, "peaks_hg38.bed")
        os.system(
            f"{self.liftover} {self.file_peaks_sort} {self.file_chain} "
            f"{file_peaks_hg38} {file_ummap}"
        )

        df_hg19 = pd.read_csv(self.file_peaks_sort, sep="\t", header=None)
        df_hg19["length"] = df_hg19.iloc[:, 2] - df_hg19.iloc[:, 1]
        len_down = np.min(df_hg19["length"]) - 20
        len_up = np.max(df_hg19["length"]) + 100

        df_hg38 = pd.read_csv(file_peaks_hg38, sep="\t", header=None)
        df_hg38["peak_hg38"] = df_hg38.apply(lambda x: f"{x[0]}-{x[1]}-{x[2]}", axis=1)
        df_hg38["length"] = df_hg38.iloc[:, 2] - df_hg38.iloc[:, 1]
        df_hg38 = df_hg38.loc[df_hg38["length"] < len_up, :]
        df_hg38 = df_hg38.loc[df_hg38["length"] > len_down, :]

        sel_peaks_hg19 = df_hg38.iloc[:, 3]
        adata_atac_out = self.adata[:, sel_peaks_hg19]
        adata_atac_out.var["peaks_hg19"] = adata_atac_out.var.index
        adata_atac_out.var.index = df_hg38["peak_hg38"]

        self.adata = adata_atac_out

    def quality_control(
        self,
        min_features: int = 1000,
        max_features: int = 60000,
        min_percent: Optional[float] = None,
        min_cells: Optional[int] = None,
    ):
        adata_atac = self.adata
        epi.pp.filter_cells(adata_atac, min_features=min_features)
        epi.pp.filter_cells(adata_atac, max_features=max_features)
        if min_percent is not None:
            by = adata_atac.obs["celltype"]
            agg_idx = (
                pd.Index(by.cat.categories)
                if pd.api.types.is_categorical_dtype(by)
                else pd.Index(np.unique(by))
            )
            agg_sum = sparse.coo_matrix(
                (
                    np.ones(adata_atac.shape[0]),
                    (agg_idx.get_indexer(by), np.arange(adata_atac.shape[0])),
                )
            ).tocsr()
            sum_x = agg_sum @ (adata_atac.X != 0)
            df_percent = (
                pd.DataFrame(
                    sum_x.toarray(), index=agg_idx, columns=adata_atac.var.index
                )
                / adata_atac.obs.value_counts("celltype")
                .loc[agg_idx]
                .to_numpy()[:, np.newaxis]
            )
            df_percent_max = np.max(df_percent, axis=0)
            sel_peaks = df_percent.columns[df_percent_max > min_percent]
            self.adata = self.adata[:, sel_peaks]
        elif min_cells is not None:
            epi.pp.filter_features(adata_atac, min_cells=min_cells)

    def deepen_atac(self, num_pc: int = 50, num_cell_merge: int = 10):
        random.seed(1234)
        adata_atac_sample_cluster = self.adata.copy()
        lsi(adata_atac_sample_cluster, n_components=num_pc)
        adata_atac_sample_cluster.obsm["X_lsi"] = adata_atac_sample_cluster.obsm[
            "X_lsi"
        ][:, 1:]
        sc.pp.neighbors(
            adata_atac_sample_cluster,
            use_rep="X_lsi",
            metric="cosine",
            n_neighbors=int(num_cell_merge),
            n_pcs=num_pc - 1,
        )

        list_atac_index = []
        list_neigh_index = []
        for cell_atac in list(adata_atac_sample_cluster.obs.index):
            cell_atac = [cell_atac]
            cell_atac_index = np.where(
                adata_atac_sample_cluster.obs.index == cell_atac[0]
            )[0]
            cell_neighbor_idx = np.nonzero(
                adata_atac_sample_cluster.obsp["connectivities"]
                .getcol(cell_atac_index)
                .toarray()
            )[0]
            if num_cell_merge >= len(cell_neighbor_idx):
                cell_sample_atac = np.hstack([cell_atac_index, cell_neighbor_idx])
            else:
                cell_sample_atac = np.hstack(
                    [
                        cell_atac_index,
                        np.random.choice(
                            cell_neighbor_idx, num_cell_merge, replace=False
                        ),
                    ]
                )
            list_atac_index.extend(
                [cell_atac_index[0] for _ in range(len(cell_sample_atac))]
            )
            list_neigh_index.append(cell_sample_atac)

        agg_sum = sparse.coo_matrix(
            (
                np.ones(len(list_atac_index)),
                (np.array(list_atac_index), np.hstack(list_neigh_index)),
            )
        ).tocsr()
        array_atac = agg_sum @ self.adata.X

        # self.adata = self.adata.copy()
        self.adata.X = None
        self.adata.X = array_atac

    def add_promoter(
        self, file_tss: str, flank_proximal: int = 2000, if_NB_MYCN: bool = False
    ):
        sc.pp.normalize_total(self.adata)
        sc.pp.log1p(self.adata)

        df_tss = pd.read_csv(file_tss, sep="\t", header=None)
        df_tss.columns = ["chrom", "tss", "symbol", "ensg_id", "strand"]
        df_tss = df_tss.drop_duplicates(subset="symbol")
        df_tss.index = df_tss["symbol"]
        df_tss["tss_start"] = df_tss["tss"] - 2000
        df_tss["tss_end"] = df_tss["tss"] + 2000
        df_tss["proximal_start"] = df_tss["tss"] - flank_proximal
        df_tss["proximal_end"] = df_tss["tss"] + flank_proximal
        file_promoter = os.path.join(self.path_process, "promoter.txt")
        file_proximal = os.path.join(self.path_process, "proximal.txt")
        df_promoter = df_tss.loc[
            :, ["chrom", "tss_start", "tss_end", "symbol", "ensg_id", "strand"]
        ]
        df_promoter.to_csv(file_promoter, sep="\t", header=False, index=False)
        df_proximal = df_tss.loc[
            :,
            ["chrom", "proximal_start", "proximal_end", "symbol", "ensg_id", "strand"],
        ]
        df_proximal.to_csv(file_proximal, sep="\t", header=False, index=False)

        self.generate_peaks_file()

        # add promoter to adata
        file_peaks_promoter = os.path.join(self.path_process, "peaks_promoter.txt")
        os.system(
            f"{self.bedtools} intersect -a {self.file_peaks_sort} -b {file_promoter} -wao "
            f"> {file_peaks_promoter}"
        )
        dict_promoter = defaultdict(list)
        with open(file_peaks_promoter, "r") as w_pro:
            for line in w_pro:
                list_line = line.strip().split("\t")
                if list_line[4] == ".":
                    continue
                gene_symbol = list_line[7]
                peak = list_line[3]
                gene_tss = df_tss.loc[gene_symbol, "tss"]
                coor_cre = (int(list_line[2]) + int(list_line[1])) / 2
                dist_gene_cre = abs(gene_tss - coor_cre)
                dict_promoter[gene_symbol].append((peak, dist_gene_cre))
        self.dict_promoter = dict_promoter
        all_genes = dict_promoter.keys()
        list_peaks_promoter = []
        list_genes_promoter = []
        for gene_symbol in all_genes:
            sub_peaks = dict_promoter[gene_symbol]
            sel_peak = ""
            min_dist = 2000
            for sub_peak in sub_peaks:
                if sub_peak[1] < min_dist:
                    sel_peak = sub_peak[0]
                    min_dist = sub_peak[1]
                if if_NB_MYCN:
                    if gene_symbol == "MYCN":
                        sel_peak = sub_peak[0]
                        print(f"MYCN added with peak: {sel_peak}")
            if sel_peak != "":
                list_peaks_promoter.append(sel_peak)
                list_genes_promoter.append(gene_symbol)

        self.all_promoter_genes = list_genes_promoter

        adata_gene_promoter = self.adata[:, list_peaks_promoter]
        adata_promoter = ad.AnnData(
            X=adata_gene_promoter.X,
            var=pd.DataFrame(
                data={"cRE_type": np.full(len(list_genes_promoter), "Promoter")},
                index=list_genes_promoter,
            ),
            obs=pd.DataFrame(index=adata_gene_promoter.obs.index),
        )

        adata_peak = self.adata.copy()
        adata_peak.obs = pd.DataFrame(index=self.adata.obs.index)
        adata_peak.var = pd.DataFrame(
            data={"node_type": np.full(adata_peak.var.shape[0], "cRE")},
            index=adata_peak.var.index,
        )
        adata_merge = ad.concat([adata_promoter, adata_peak], axis=1)
        self.adata_merge = adata_merge

        # proximal regulation
        file_peaks_proximal = os.path.join(self.path_process, "peaks_proximal.txt")
        os.system(
            f"{self.bedtools} intersect -a {self.file_peaks_sort} -b {file_proximal} -wao "
            f"> {file_peaks_proximal}"
        )
        dict_proximal = defaultdict(list)
        with open(file_peaks_proximal, "r") as w_pro:
            for line in w_pro:
                list_line = line.strip().split("\t")
                if list_line[4] == ".":
                    continue
                gene_symbol = list_line[7].strip().split("<-")[0]
                peak = list_line[3]
                dict_proximal[gene_symbol].append(peak)
        self.dict_promoter = dict_proximal

        all_genes = dict_proximal.keys()
        list_peaks_proximal = []
        list_genes_proximal = []
        for gene_symbol in all_genes:
            sub_peaks = dict_proximal[gene_symbol]
            list_genes_proximal.extend([gene_symbol for _ in range(len(sub_peaks))])
            list_peaks_proximal.extend(sub_peaks)
        self.all_proximal_genes = set(list_genes_proximal)

        self.df_gene_peaks = pd.DataFrame(
            {"gene": list_genes_proximal, "peak": list_peaks_proximal}
        )
        self.df_proximal = pd.DataFrame(
            {
                "region1": list_genes_proximal,
                "region2": list_peaks_proximal,
                "type": ["proximal"] * len(list_peaks_proximal),
            }
        )
        set_gene = set(self.df_rna.columns).intersection(self.all_promoter_genes)
        self.df_proximal = self.df_proximal.loc[
            self.df_proximal["region1"].apply(lambda x: x in set_gene), :
        ]

        return

    def build_graph(
        self, path_interaction: str, file_suffix: str, sel_interaction: str = "PO"
    ):
        file_pp = os.path.join(path_interaction, "PP" + file_suffix + ".txt")
        file_po = os.path.join(path_interaction, "PO" + file_suffix + ".txt")
        if sel_interaction == "PP" or sel_interaction == "ALL":
            df_pp_pre = pd.read_csv(file_pp, sep="\t", header=None)
            df_pp_pre = df_pp_pre.loc[
                df_pp_pre.apply(
                    lambda x: x.iloc[0] in self.all_promoter_genes
                    and x.iloc[1] in self.all_promoter_genes,
                    axis=1,
                ),
                :,
            ]
            df_pp_pre.columns = ["region1", "gene"]
            df_gene_peaks = self.df_gene_peaks.copy()
            df_gene_peaks.columns = ["gene", "region2"]
            df_pp = pd.merge(left=df_pp_pre, right=df_gene_peaks, on="gene")
            df_pp = df_pp.loc[:, ["region1", "region2"]]
        if sel_interaction == "PO" or sel_interaction == "ALL":
            file_po_peaks = os.path.join(self.path_process, "peaks_PO.bed")
            os.system(
                f"{self.bedtools} intersect -a {self.file_peaks_sort} -b {file_po} -wao "
                f"> {file_po_peaks}"
            )
            list_dict = []
            with open(file_po_peaks, "r") as r_po:
                for line in r_po:
                    list_line = line.strip().split("\t")
                    peak = list_line[3]
                    gene_symbol = list_line[8]
                    if gene_symbol in self.all_promoter_genes:
                        list_dict.append({"region1": gene_symbol, "region2": peak})
            df_po = pd.DataFrame(list_dict)
        if sel_interaction == "PP":
            df_interaction = df_pp
        elif sel_interaction == "PO":
            df_interaction = df_po
        elif sel_interaction == "ALL":
            df_interaction = pd.concat([df_pp, df_po])
        else:
            print("Error: please set correct parameter 'sel_interaction'! ")
            return

        self.df_distal = df_interaction.drop_duplicates()
        self.df_distal["type"] = ["distal"] * self.df_distal.shape[0]
        set_gene = set(self.df_rna.columns)
        self.df_distal = self.df_distal.loc[
            self.df_distal["region1"].apply(lambda x: x in set_gene), :
        ]
        self.df_graph = pd.concat([self.df_proximal, self.df_distal], axis=0)

        return

    def add_eqtl(self, file_eqtl: str):
        file_eqtl_peaks = os.path.join(self.path_process, "peaks_eQTL.bed")
        os.system(
            f"{self.bedtools} intersect -a {self.file_peaks_sort} -b {file_eqtl} -wao "
            f"> {file_eqtl_peaks}"
        )
        list_dict_eqtl = []
        with open(file_eqtl_peaks, "r") as r_po:
            for line in r_po:
                list_line = line.strip().split("\t")
                peak = list_line[3]
                gene_symbol = list_line[8]
                if gene_symbol in self.all_promoter_genes:
                    list_dict_eqtl.append({"region1": gene_symbol, "region2": peak})
        df_eqtl = pd.DataFrame(list_dict_eqtl)
        df_eqtl = df_eqtl.drop_duplicates()
        df_eqtl["type"] = ["eQTL"] * df_eqtl.shape[0]
        self.df_eqtl = df_eqtl
        set_gene = set(self.df_rna.columns)
        self.df_eqtl = self.df_eqtl.loc[
            self.df_eqtl["region1"].apply(lambda x: x in set_gene), :
        ]
        self.df_graph = pd.concat([self.df_graph, self.df_eqtl], axis=0)
        self.df_graph = self.df_graph.drop_duplicates(
            subset=["region1", "region2"], keep="first"
        )

    def build_tf_graph(self, file_tf: str):
        df_tf = pd.read_csv(file_tf, sep="\t", header=None)
        df_tf = df_tf.iloc[:, :2]
        df_tf.columns = ["TF", "TargetGene"]
        set_gene = set(self.df_rna.columns).intersection(self.all_promoter_genes)

        tf_base_filtered = df_tf[
            df_tf["TF"].isin(set_gene) & df_tf["TargetGene"].isin(set_gene)
        ]
        connections = [pair for pair in itertools.product(set_gene, set_gene)]
        gene_pair_base = connections
        tf_map_gene = set(tf_base_filtered["TF"].unique())
        target_map_gene = set(tf_base_filtered["TargetGene"].unique())
        tf_base_tuples = set(
            zip(tf_base_filtered["TF"], tf_base_filtered["TargetGene"])
        )
        map_pair = tf_base_tuples.intersection(gene_pair_base)
        map_pair_list = list(map_pair)

        df_tf = pd.DataFrame(map_pair_list, columns=["TF", "TargetGene"])
        df_tf_self = pd.DataFrame(
            {"TF": sorted(list(set_gene)), "TargetGene": sorted(list(set_gene))}
        )
        self.df_tf = pd.concat([df_tf_self, df_tf], axis=0)

    def build_tf_graph1(self, df_tf):
        set_gene = set(self.df_rna.columns).intersection(self.all_promoter_genes)

        tf_base_filtered = df_tf[
            df_tf["TF"].isin(set_gene) & df_tf["TargetGene"].isin(set_gene)
        ]
        connections = [pair for pair in itertools.product(set_gene, set_gene)]
        gene_pair_base = connections
        tf_map_gene = set(tf_base_filtered["TF"].unique())
        target_map_gene = set(tf_base_filtered["TargetGene"].unique())
        tf_base_tuples = set(
            zip(tf_base_filtered["TF"], tf_base_filtered["TargetGene"])
        )
        map_pair = tf_base_tuples.intersection(gene_pair_base)
        map_pair_list = list(map_pair)

        df_tf = pd.DataFrame(map_pair_list, columns=["TF", "TargetGene"])
        # df_tf_self = pd.DataFrame({'TF': sorted(list(set_gene)), 'TargetGene': sorted(list(set_gene))})
        # self.df_tf = pd.concat([df_tf_self, df_tf], axis=0)
        self.df_tf = df_tf

    def generate_data_list(self):
        graph_data = self.df_graph
        graph_tf = self.df_tf
        adata_atac = self.adata
        adata_merge = self.adata_merge
        all_cre_gene = set(graph_data["region1"]).union(set(graph_data["region2"]))
        all_tf_gene = set(graph_tf["TF"]).union(set(graph_tf["TargetGene"]))
        all_peaks = all_cre_gene.union(all_tf_gene)
        adata_merge_peak = adata_merge[
            :, [one_peak for one_peak in adata_merge.var.index if one_peak in all_peaks]
        ]
        array_peak = np.array(adata_merge_peak.var.index)
        list_gene_peak = [one_peak for one_peak in array_peak if one_peak[:3] != "chr"]
        self.df_rna = self.df_rna.loc[:, list_gene_peak]
        self.df_rna = self.df_rna / np.array(np.sum(self.df_rna, axis=1))[:, np.newaxis]
        array_celltype = np.unique(np.array(adata_atac.obs["celltype"]))
        # cRE-Gene
        array_region1 = graph_data["region1"].apply(
            lambda x: np.argwhere(array_peak == x)[0, 0]
        )
        array_region2 = graph_data["region2"].apply(
            lambda x: np.argwhere(array_peak == x)[0, 0]
        )
        df_graph_index = torch.tensor(
            [np.array(array_region2), np.array(array_region1)], dtype=torch.int64
        )
        # TF-Gene
        array_tf = graph_tf["TF"].apply(lambda x: np.argwhere(array_peak == x)[0, 0])
        array_target = graph_tf["TargetGene"].apply(
            lambda x: np.argwhere(array_peak == x)[0, 0]
        )
        df_graph_tf = torch.tensor(
            [np.array(array_tf), np.array(array_target)], dtype=torch.int64
        )
        df_merge_peak = adata_merge_peak.to_df()
        list_graph = []
        for i_cell in range(0, adata_atac.n_obs):
            one_cell = adata_atac.obs.index[i_cell]
            label = adata_atac.obs.loc[one_cell, "celltype"]
            label_rna = adata_atac.obs.loc[one_cell, "celltype_rna"]
            label_exp = self.df_rna.loc[label_rna, list_gene_peak].tolist()
            label_idx = torch.tensor(
                np.argwhere(array_celltype == label)[0], dtype=torch.int64
            )
            cell_data = Data(
                x=torch.reshape(
                    torch.Tensor(df_merge_peak.loc[one_cell, :]),
                    (adata_merge_peak.shape[1], 1),
                ),
                edge_index=df_graph_index,
                edge_tf=df_graph_tf.T,
                y=label_idx,
                y_exp=torch.tensor(label_exp),
                cell=one_cell,
            )
            list_graph.append(cell_data)

        self.list_graph = list_graph
        self.array_peak = array_peak
        self.array_celltype = array_celltype

        return


class ATACGraphDataset(InMemoryDataset):
    def __init__(self, root: str, data_list: List = None):
        self.data_list = data_list
        super(ATACGraphDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["some_file_1"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = self.data_list
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def prepare_model_input(
    adata_atac,
    path_data_root: str,
    file_atac: str,
    df_rna_celltype: DataFrame,
    path_eqtl: str,
    Hi_C_file_suffix: str = "",
    min_features: Optional[float] = None,
    max_features: Optional[float] = None,
    min_percent: Optional[float] = 0.05,
    hg19tohg38: bool = False,
    if_NB: bool = False,
    deepen_data: bool = True,
    use_additional_tf=False,
    tissue_cuttof=3,
):

    if not os.path.exists(path_data_root):
        os.mkdir(path_data_root)

    # Try to find data files - user should provide path or set environment variable
    # For now, we'll try relative path first, then check environment variable
    basepath = os.path.abspath(__file__)
    package_dir = os.path.dirname(
        os.path.dirname(basepath)
    )  # Go up to scregat package dir
    data_dir = os.environ.get("SCREGAT_DATA_DIR", os.path.join(package_dir, "data"))
    file_chrom_hg38 = os.path.join(data_dir, "hg38.chrom.sizes")

    print("only dataset_obj ...")

    dataset_ATAC = ATACDataset(
        adata_atac=adata_atac,
        raw_filename=file_atac,
        data_root=path_data_root,
        file_chrom=file_chrom_hg38,
    )

    # dataset_ATAC.adata.obs['celltype'] = dataset_ATAC.adata.obs['seurat_annotations']
    if hg19tohg38:
        dataset_ATAC.hg19tohg38()
    vec_num_feature = np.array(np.sum(dataset_ATAC.adata.X != 0, axis=1))
    if min_features is None:
        default_min = int(np.percentile(vec_num_feature, 1))
        min_features = default_min
    if max_features is None:
        default_max = int(np.percentile(vec_num_feature, 99))
        max_features = default_max
    dataset_ATAC.quality_control(
        min_features=min_features, max_features=max_features, min_percent=min_percent
    )

    # deep atac
    if deepen_data:
        dataset_ATAC.deepen_atac(num_cell_merge=10)

    # add RNA-seq data
    dataset_ATAC.df_rna = df_rna_celltype

    file_gene_hg38 = os.path.join(data_dir, "genes.protein.tss.tsv")
    dataset_ATAC.add_promoter(file_gene_hg38, if_NB_MYCN=if_NB)
    if if_NB:
        print(dataset_ATAC.df_rna.loc[:, "MYCN"])

    # Hi-C
    print("processing Hi-C ...")
    path_hic = data_dir
    dataset_ATAC.build_graph(
        path_hic, file_suffix=Hi_C_file_suffix, sel_interaction="ALL"
    )
    dataset_ATAC.add_eqtl(path_eqtl)
    # df_graph_PLAC = dataset_ATAC.df_graph

    # TF
    print("processing TF ...")
    path_tf = os.path.join(data_dir, "trrust_rawdata.human.tsv")
    path_tf2 = os.path.join(data_dir, "TF_Gene_tissue_cutoff1.csv")
    if use_additional_tf:
        print("additional TF...")
        tf_base = pd.read_csv(path_tf2, index_col=0)
        tf_base = tf_base[tf_base.tissue_count > tissue_cuttof]
        tf_base = tf_base.iloc[:, :2]
        tf_base.columns = ["TF", "TargetGene"]
        df_tf = pd.read_csv(path_tf, sep="\t", header=None)
        df_tf = df_tf.iloc[:, :2]
        df_tf.columns = ["TF", "TargetGene"]
        df_tf = pd.concat([df_tf, tf_base])
        print("total candidate tf-gene: ", len(df_tf))
        dataset_ATAC.build_tf_graph1(df_tf)
    else:
        dataset_ATAC.build_tf_graph(path_tf)
    # df_graph_TF = dataset_ATAC.df_tf

    dataset_ATAC.generate_data_list()
    return dataset_ATAC
