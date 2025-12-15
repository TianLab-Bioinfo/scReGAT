import os
import sys
import argparse
import random
import pickle
import numpy as np
import pandas as pd
import torch
import anndata as ad
from tqdm import tqdm # 新增 tqdm 用于显示进度


try:
    from scregat.model import *
    from scregat.train import *
    from scregat.data_process import prepare_model_input, sum_counts, plot_edge, ATACGraphDataset
except ImportError:
    print("Error: Could not import 'run_scReGAT' or 'scregat'. Please ensure you are in the correct directory structure.")
    sys.exit(1)

# 尝试导入 scanpy，用于读取 RNA 数据
try:
    import scanpy as sc
except ImportError:
    sc = None

def parse_args():
    parser = argparse.ArgumentParser(description="Run scReGAT Model Training and Inference")

    # --- I/O 参数 ---
    parser.add_argument('--input_file', type=str, required=True, 
                        help='Path to the input pickle file (e.g., dataset_atac_core_MFG.pkl)')
    parser.add_argument('--output_file', type=str, required=True, 
                        help='Path to save the output AnnData file (e.g., result.h5ad)')
    
    # --- 单细胞表达谱整合参数 (新增) ---
    parser.add_argument('--use_sc_exp', action='store_true',
                        help='If set, integrate single-cell RNA expression data into the graph.')
    parser.add_argument('--rna_file', type=str, default=None,
                        help='Path to the RNA .h5ad file. Required if --use_sc_exp is set.')

    # --- 模型保存与加载 ---
    parser.add_argument('--save_model_path', type=str, default=None, 
                        help='(Optional) Path to save the trained model parameters')
    parser.add_argument('--load_model_path', type=str, default=None, 
                        help='(Optional) Path to load a pre-trained model checkpoint')
    
    # --- 训练控制 ---
    parser.add_argument('--skip_train', action='store_true',
                        help='If set, skip the training phase and run inference directly')
    parser.add_argument('--seed', type=int, default=4446, help='Random seed (default: 4446)')
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=15, help='Training batch size (default: 15)')
    parser.add_argument('--sparse_loss_weight', type=float, default=0.1, help='Weight for sparse loss (default: 0.1)')
    
    # --- 测试/推断参数 ---
    parser.add_argument('--test_batch_size', type=int, default=20, help='Inference batch size (default: 20)')
    parser.add_argument('--test_ratio', type=float, default=0.5, help='Ratio of cells to use for testing (default: 0.5)')
    
    # --- 硬件参数 ---
    parser.add_argument('--gpu', type=int, default=1, help='GPU ID to use. Use -1 for CPU. (default: 1)')

    return parser.parse_args()

def set_global_seed(seed, use_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed) 

def main():
    args = parse_args()

    # 1. 设置设备
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        use_cuda = True
        print(f"Using GPU: cuda:{args.gpu}")
    else:
        device = torch.device('cpu')
        use_cuda = False
        print("Using CPU")

    # 2. 设置随机种子
    set_global_seed(args.seed, use_cuda)

    # 3. 加载 ATAC Pickle 数据
    print(f"Loading ATAC data from {args.input_file}...")
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
        
    with open(args.input_file, 'rb') as f:
        dataset_atac = pickle.load(f)

    # 3.5. (新增) 处理单细胞 RNA 表达数据整合逻辑
    if args.use_sc_exp:
        print("\n[Mode] Single-cell Expression Integration: ENABLED")
        if sc is None:
            raise ImportError("Scanpy is required for --use_sc_exp. Please install it (pip install scanpy).")
        
        if not args.rna_file:
            raise ValueError("--rna_file path is required when --use_sc_exp is set.")
        
        if not os.path.exists(args.rna_file):
            raise FileNotFoundError(f"RNA file not found: {args.rna_file}")

        print(f"Loading RNA AnnData from {args.rna_file}...")
        rna_adata = sc.read_h5ad(args.rna_file)
        
        # --- 下方逻辑基于你提供的代码段 ---
        print("Preprocessing RNA data and aligning with ATAC graph...")
        # 去除重复变量
        rna_adata = rna_adata[:, ~rna_adata.var_names.duplicated()]

        # 获取基因列表，过滤掉以 'chr' 开头的基因
        # dataset_atac.array_peak 包含了 peaks (chr开头) 和 genes (其他)
        gene_list = [t for t in dataset_atac.array_peak if not t.startswith('chr')]
        
        # 检查 RNA 数据中是否涵盖了所有需要的基因
        missing_genes = set(gene_list) - set(rna_adata.var_names)
        if len(missing_genes) > 0:
            print(f"Warning: {len(missing_genes)} genes from ATAC graph not found in RNA data. (e.g., {list(missing_genes)[:3]}...)")
            # 注意：如果缺失严重，下面的切片可能会报错或行为异常，这里假设用户数据是匹配的
        
        # 将基因列表转换为 numpy 数组
        gene_list = np.array(gene_list)

        # 筛选 rna_adata 中的基因
        # 注意：这里需要确保 rna_adata 包含 gene_list 中的基因，否则 scanpy/pandas 会报错
        rna_adata = rna_adata[:, gene_list]

        # 获取细胞列表
        cells = [t.cell for t in dataset_atac.list_graph]

        # 检查细胞对齐情况
        missing_cells = set(cells) - set(rna_adata.obs_names)
        if len(missing_cells) > 0:
            raise ValueError(f"Error: {len(missing_cells)} cells in ATAC graph are missing from RNA data. Data must be aligned.")

        # 获取表达矩阵并按照 ATAC 图中的细胞顺序排列
        # .toarray() 确保是 dense 格式，方便计算
        exp_mt = rna_adata[cells, :].X
        if hasattr(exp_mt, 'toarray'):
            exp_mt = exp_mt.toarray()
        elif hasattr(exp_mt, 'todense'): # 兼容 matrix 类型
            exp_mt = np.array(exp_mt.todense())

        print("Injecting expression data into graph nodes...")
        single_graph = []
        for idx, t in tqdm(enumerate(dataset_atac.list_graph), total=len(dataset_atac.list_graph), desc="Updating Graphs"):
            # 防止除以 0
            row_sum = np.sum(exp_mt[idx, :])
            if row_sum > 0:
                normalized_exp = exp_mt[idx, :] / row_sum
            else:
                normalized_exp = exp_mt[idx, :] # 全0保持全0
            
            t.y_exp = torch.tensor(normalized_exp, dtype=torch.float32) 
            single_graph.append(t)
        
        # 更新数据集中的图列表
        dataset_atac.list_graph = single_graph
        print("Integration completed.\n")
    else:
        print("\n[Mode] Single-cell Expression Integration: DISABLED")
        # 如果不使用，scReGAT 可能会使用默认的统一分布或先验，具体取决于模型内部实现
        pass

    # 3.6. (新增) 报告边/节点信息
    print("="*40)
    print("       DATASET STATISTICS REPORT       ")
    print("="*40)
    all_nodes = dataset_atac.array_peak
    genes = [t for t in all_nodes if not t.startswith('chr')]
    peaks = [t for t in all_nodes if t.startswith('chr')]
    
    print(f"Total Cells (Graphs): {len(dataset_atac.list_graph)}")
    print(f"Total Nodes:          {len(all_nodes)}")
    print(f"  - Genes (Targets):  {len(genes)}")
    print(f"  - Peaks (Regions):  {len(peaks)}")
    
    if args.use_sc_exp:
        print(f"Expression Feature:   Using scRNA-seq (y_exp injected)")
    else:
        print(f"Expression Feature:   Not using external scRNA-seq")
    print("="*40 + "\n")


    # 4. 划分训练/测试集
    cells = dataset_atac.adata.obs_names.values
    test_size = int(len(cells) * args.test_ratio)
    
    # 确保 test_cell 在 cells 列表中
    test_cell = random.sample(list(cells), test_size)
    print(f"Total cells: {len(cells)}, Test cells: {len(test_cell)}")

    # 5. 预处理图数据
    print("Preprocessing graph for model...")
    balanced_train_graph, test_graph = preprocess_graph(
        dataset_atac, 
        test_cell, 
        max_random_sample=0, 
        seed=args.seed, 
        if_test=False,
        is_balances=False
    )

    # 6. 初始化模型
    model = SCReGAT()
    model.to(device) 

    # 7. (可选) 加载预训练模型
    if args.load_model_path:
        print(f"Loading pre-trained model from {args.load_model_path}...")
        if not os.path.exists(args.load_model_path):
            raise FileNotFoundError(f"Model file not found: {args.load_model_path}")
        
        state_dict = torch.load(args.load_model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")

    # 8. 训练 (如果未设置 skip_train)
    if not args.skip_train:
        print(f"Starting training for {args.epochs} epochs...")
        model = train_model(
            model, 
            balanced_train_graph, 
            num_epoch=args.epochs, 
            batch_size=args.batch_size, 
            lr=args.lr, 
            max_grad_norm=1.0, 
            sparse_loss_weight=args.sparse_loss_weight,
            if_zero=False,
            use_device=device
        )
        
        if args.save_model_path:
            print(f"Saving trained model to {args.save_model_path}...")
            model_dir = os.path.dirname(args.save_model_path)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(model.state_dict(), args.save_model_path)
            print("Model saved.")
    else:
        print("Skipping training phase (--skip_train is set).")

    # 9. 测试/推断
    print("Running inference...")
    res_edge, test_barcodes = test_model(
        dataset_atac, 
        model, 
        test_graph, 
        batch_size=args.test_batch_size, 
        device=device, 
        if_test=False
    )

    # 10. 构建结果 AnnData
    print("Constructing result AnnData...")
    # 确保 test_barcodes 与 obs 索引对齐
    obs_subset = dataset_atac.adata.obs.loc[test_barcodes, :]
    adata_edge = ad.AnnData(X=res_edge, obs=obs_subset)
    
    edge_info = get_edge_info(dataset_atac)
    edge_info['peak'] = edge_info['peak'].astype(str)
    edge_info['gene'] = edge_info['gene'].astype(str)
    edge_info['edge'] = edge_info['peak'] + "_" + edge_info['gene']
    
    adata_edge.var_names = edge_info.edge.values

    # 11. 保存结果
    print(f"Saving results to {args.output_file}...")
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    adata_edge.write(args.output_file)
    print("All tasks completed.")

if __name__ == "__main__":
    main()