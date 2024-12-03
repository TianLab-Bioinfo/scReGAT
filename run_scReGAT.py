# 基础库
import os
import numpy as np
import pandas as pd
import pickle
import random
from time import time
from collections import defaultdict
from typing import Optional, Mapping, List, Union

import anndata as ad
from anndata import AnnData
from scipy import sparse
from statsmodels.distributions.empirical_distribution import ECDF
import sklearn

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

import torch
import torch.nn as nn
import torch.nn.functional as F

import scanpy as sc
import episcanpy.api as epi
import cosg

from scregat.data_process import prepare_model_input, sum_counts, plot_edge, ATACGraphDataset

import torch
import random
import numpy as np
from tqdm import tqdm

import pandas as pd
import torch
import anndata as ad
from torch_geometric.loader import DataLoader
import scanpy as sc
from scregat.data_process import prepare_model_input, sum_counts, plot_edge, ATACGraphDataset
import numpy as np
import pickle
import random
from tqdm import tqdm



class EdgeDiversityLoss1(nn.Module):
    def __init__(self, diversity_weight=1.0):
        """
        自定义多样性损失函数
        :param diversity_weight: 控制熵惩罚的权重
        """
        super(EdgeDiversityLoss1, self).__init__()
        self.diversity_weight = diversity_weight

    def forward(self, edge_weights):
        prob_distribution = torch.softmax(edge_weights / 0.8, dim=0)
        entropy_loss = -self.diversity_weight * torch.sum(prob_distribution * torch.log(prob_distribution + 1e-6), dim=0)
        
        # 添加均匀性损失
        target_distribution = torch.full_like(prob_distribution, 1.0 / edge_weights.numel())
        uniformity_loss = torch.mean((prob_distribution - target_distribution) ** 2)

        # 总损失
        loss = torch.mean(entropy_loss) +  uniformity_loss
        return loss



class SCReGAT(torch.nn.Module):
    def __init__(self,
                 node_input_dim=1,
                 node_output_dim=8,
                 edge_embedding_dim=8,
                 hidden_channels=16,  # 增加 hidden_channels
                 gat_input_channels=8,
                 gat_hidden_channels=8,  # 增加 GAT 隐藏通道
                 seq_dim=768,
                 seq2node_dim=1,
                 max_tokens=1024,
                 dropout=0.1,  # 增加 dropout
                 num_head_1=16,  # 增加 GAT 头数
                 num_head_2=16):  # 增加 GAT 头数
        super(SCReGAT, self).__init__()

        # Sequence transformation layer
        self.NN_seq = nn.Sequential(
            nn.Linear(seq_dim, 512),
            nn.BatchNorm1d(512),  # BatchNorm for sequence layer
            nn.LeakyReLU(negative_slope=0.01),  # LeakyReLU activation
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),  # BatchNorm for sequence layer
            nn.LeakyReLU(negative_slope=0.01),  # LeakyReLU activation
            nn.Linear(128, seq2node_dim)
        )

        # Node feature transformation with BatchNorm
        self.NN_node = nn.Sequential(
            nn.Linear(node_input_dim, 128),
            nn.BatchNorm1d(128),  # BatchNorm for node layer
            nn.LeakyReLU(negative_slope=0.01),  # LeakyReLU activation
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  # BatchNorm for node layer
            nn.LeakyReLU(negative_slope=0.01),  # LeakyReLU activation
            nn.Linear(64, node_output_dim)
        )

        # Edge feature transformation with BatchNorm
        self.NN_edge = nn.Sequential(
            nn.Linear(3, 32, bias=True),
            nn.BatchNorm1d(32),  # BatchNorm for edge layer
            nn.LeakyReLU(negative_slope=0.01),  # LeakyReLU activation
            nn.Linear(32, 16, bias=True),
            nn.BatchNorm1d(16),  # BatchNorm for edge layer
            nn.LeakyReLU(negative_slope=0.01),  # LeakyReLU activation
            nn.Linear(16, edge_embedding_dim, bias=True),
            nn.BatchNorm1d(edge_embedding_dim),  # BatchNorm for edge layer
            nn.LeakyReLU(negative_slope=0.01)   # LeakyReLU activation
        )


        # GAT layers with increased heads and self-loops
        self.NN_conv1 = GATConv(node_output_dim, hidden_channels, heads=num_head_1, dropout=dropout, edge_dim=edge_embedding_dim, add_self_loops=False)
        self.NN_flatten1 = nn.Linear(num_head_1 * hidden_channels, hidden_channels)

        self.NN_conv2 = GATConv(hidden_channels, hidden_channels, heads=num_head_2, dropout=dropout, add_self_loops=False)
        self.NN_flatten2 = nn.Linear(num_head_2 * hidden_channels, hidden_channels)

        self.dropout = nn.Dropout(0.1)  
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU(negative_slope=0.01)


    def forward(self, seq_data, raw_x, edge_index, edge_tf, batch, gene_num, gene_id_vec, is_test=False):
        data = raw_x
        p_batch = batch.unique()
        data = self.NN_node(data)
        hidden_edge_input = torch.cat((raw_x[edge_index[0]] * raw_x[edge_index[1]], raw_x[edge_index[0]], raw_x[edge_index[1]]), dim=1)
        self.data = data
        hidden_edge = self.NN_edge(hidden_edge_input).tanh()
        self.model_edge = hidden_edge
        self.edge = torch.median(hidden_edge, dim=1)[0]
        data, atten_w1 = self.NN_conv1(data, edge_index, edge_attr=hidden_edge, return_attention_weights=True)
        data_1 = self.leaky(self.NN_flatten1(data))
        data_2, atten_w2 = self.NN_conv2(data_1, edge_tf, return_attention_weights=True)
        data_2 = self.leaky(self.NN_flatten2(data_2))
        data = data_1 + data_2
        gene_out = -F.log_softmax(data[gene_id_vec==1], dim=1)[:, 0]
        return gene_out, atten_w1


# 设置随机数种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, train_graph, num_epoch=5, batch_size=12, lr=0.0001, max_grad_norm=1.0, sparse_loss_weight=0.1, use_device='cuda:0'):
    device = use_device if torch.cuda.is_available() else 'cpu'
    model.to(device)
    loss_exp = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_sparse1 = EdgeDiversityLoss1(diversity_weight=1)

    for epoch in range(num_epoch):
        train_loader = DataLoader(train_graph, batch_size=batch_size, shuffle=True)
        model.train()
        running_loss = 0.0
        running_loss1 = 0.0
        running_sparse_loss = 0.0 
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}", unit="batch")
        for idx, sample in enumerate(progress_bar):
            gene_num = sample.y_exp.shape[0]
            optimizer.zero_grad()
            gene_pre, atten = model(
                sample.seq_data.to(device),
                sample.x.to(device),
                sample.edge_index.to(device),
                sample.edge_tf.T.to(device),
                sample.batch.to(device), 
                gene_num, 
                sample.id_vec.to(device)
            )
            edge_temp = model.edge
            labels = sample.y.to(device)
            index = torch.where(sample.x[sample.id_vec == 1] > 0)[0]
            loss1 = loss_exp(gene_pre.flatten()[index], sample.y_exp.to(device)[index]) 
            temp_var_edge = torch.stack(torch.split(edge_temp, sample.edge_index.shape[1] // len(sample.y)))
            loss2 = sparse_loss_weight * criterion_sparse1(temp_var_edge) 
            loss = loss1 + loss2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            running_loss += loss.item()
            running_loss1 += loss1.item()
            running_sparse_loss += loss2.item()  

            progress_bar.set_postfix(
                loss=running_loss / (progress_bar.n + 1),
                loss1=running_loss1 / (progress_bar.n + 1),
                sparse_loss=running_sparse_loss / (progress_bar.n + 1)
            )
            torch.cuda.empty_cache()
    
        print(f"Epoch [{epoch+1}/{num_epoch}], Average Loss: {running_loss / len(train_loader):.4f}, "
              f"Loss1: {running_loss1 / len(train_loader):.4f}, Sparse Loss: {running_sparse_loss / len(train_loader):.4f}")
    return model

def test_model(model, test_graph, batch_size=1, device='cuda:2'):
    """
    运行模型测试过程并收集结果数据。

    参数：
    - model: 要测试的模型
    - test_graph: 测试数据集
    - loss_exp: 损失函数
    - batch_size: 批次大小
    - device: 设备 ('cuda:2' 或 'cpu')

    返回：
    - result_dict: 包含结果数据的字典
    """
    test_loader = DataLoader(test_graph, batch_size=batch_size, shuffle=False, pin_memory=True)
    model.eval()
    model.to(device)

    cell_type = []
    test_barcodes = []
    cell_link_atten = []
    cell_link_activity = []
    cell_link_edge = []
    cell_link_enhancer = []

    with torch.no_grad():
        for idx, sample in tqdm(enumerate(test_loader), total=len(test_loader), desc="Processing Batches"):
            gene_num = sample.y_exp.shape[0]
            gene_pre, atten = model(
                sample.seq_data.to(device),
                sample.x.to(device), 
                sample.edge_index.to(device), 
                sample.edge_tf.T.to(device), 
                sample.batch.to(device), 
                gene_num, 
                sample.id_vec.to(device), 
                is_test=True
            )
            sample_size = len(sample.y)
            test_barcodes.extend(sample.cell)  # 确保在 CPU 上
            cell_type.extend(sample.y.cpu().numpy())

            # 处理 Edge
            edge_temp = model.edge.flatten().to('cpu')
            edge_lists = edge_temp.split(edge_temp.size(0) // sample_size)

            for edge in edge_lists:
                cell_link_edge.append(edge)

            # 处理注意力数据
            atten_x_indices = atten[0][1].cpu()
            atten_enhancer_indices = atten[0][0].cpu()
            flattened_sample_x = sample.x[atten_x_indices].flatten().to('cpu')
            flattened_sample_enhancer = sample.x[atten_enhancer_indices].flatten().to('cpu')
            atten1_max = atten[1].mean(dim=1).to('cpu')

            # 分割并收集数据
            for x, enhancer, atten_ in zip(
                flattened_sample_x.split(flattened_sample_x.size(0) // sample_size),
                flattened_sample_enhancer.split(flattened_sample_enhancer.size(0) // sample_size),
                atten1_max.split(atten1_max.size(0) // sample_size)
            ):
                cell_link_activity.append(x)
                cell_link_atten.append(atten_)
                cell_link_enhancer.append(enhancer)

            # 清理缓存
            torch.cuda.empty_cache()

    cell_link_atten = torch.stack(cell_link_atten)
    cell_link_edge = torch.stack(cell_link_edge)
    cell_link_activity = torch.stack(cell_link_activity)
    cell_link_enhancer = torch.stack(cell_link_enhancer)
    result_dict = {
        'cell_type': cell_type,
        'test_barcodes': test_barcodes,
        'cell_link_atten': cell_link_atten,
        'cell_link_activity': cell_link_activity,
        'cell_link_edge': cell_link_edge,
        'cell_link_enhancer': cell_link_enhancer
    }

    return result_dict


def process_samples(dataset_atac, dataset_graph, test_cell):
    """
    处理样本数据，将其分为训练集和测试集。

    参数：
    - dataset_atac: ATAC-seq 数据集
    - dataset_graph: 图数据集
    - test_cell: 测试细胞列表

    返回：
    - train_graph: 训练集图数据
    - test_graph: 测试集图数据
    - train_cell_type: 训练集细胞类型
    - test_cell_type: 测试集细胞类型
    """

    data = dataset_atac.array_peak
    torch_vector = torch.zeros(len(data))
    for idx, item in enumerate(data):
        if item.startswith('chr'):
            torch_vector[idx] = 0 
        else:
            torch_vector[idx] = 1 

    train_graph = []
    test_graph = []
    test_cell_type = []
    train_cell_type = []

    for i, sample in tqdm(enumerate(dataset_graph), total=len(dataset_graph), desc='Processing samples'):
        sample.id_vec = torch_vector
        sample.seq_data = 1
        if sample.cell in test_cell:
            test_graph.append(sample)
            test_cell_type.append(sample.y.item())
        else:
            train_graph.append(sample)
            train_cell_type.append(sample.y.item())

    return train_graph, test_graph, train_cell_type, test_cell_type


def process_samples_add_random(graph, torch_vector, max_samples=1000):
    """
    处理图数据样本，将其转换并存储在训练集中。

    参数：
    - graph: 图数据集
    - torch_vector: 用于标识的向量
    - max_samples: 最大处理样本数

    返回：
    - train_graph: 训练集图数据
    - train_cell_type: 训练集细胞类型
    """
    dataset_graph = graph.copy()
    dataset_graph.shuffle()
    train_graph = []
    train_cell_type = []

    for i, sample in tqdm(enumerate(dataset_graph), total=min(len(dataset_graph), max_samples), desc='Processing samples'):
        sample.id_vec = torch_vector
        sample.seq_data = 2
        sample.x = torch.rand_like(sample.x) * 3
        sample.y_exp = sample.y_exp[torch.randperm(sample.y_exp.size(0))]
        sample.y = torch.tensor([6])
        train_graph.append(sample)
        train_cell_type.append(sample.y.item())
        if i == max_samples - 1:
            break
    return train_graph, train_cell_type

import torch
import random
from collections import Counter
from tqdm import tqdm

def balance_classes(train_graph, train_cell_type, seed=42):
    """
    平衡训练集中的类别样本数量。

    参数：
    - train_graph: 训练集图数据
    - train_cell_type: 训练集细胞类型
    - seed: 随机数种子，默认值为42

    返回：
    - balanced_train_graph: 类别平衡的训练集图数据
    - balanced_train_cell_type: 类别平衡的训练集细胞类型
    """
    
    # 设置随机数种子
    random.seed(seed)
    torch.manual_seed(seed)

    # 计算每个类别的样本数量
    class_counts = Counter(train_cell_type)

    # 找到最多的类别样本数
    max_count = max(class_counts.values())

    # 复制样本以实现类别平衡
    balanced_train_graph = []

    for cell_type, count in class_counts.items():
        # 找到对应的样本
        samples = [sample for sample in train_graph if sample.y.item() == cell_type]

        # 计算需要复制的样本数量
        samples_to_add = max_count - count

        # 复制样本并添加到平衡后的训练图中
        balanced_train_graph.extend(samples)
        balanced_train_graph.extend(random.choices(samples, k=samples_to_add))

    # 更新细胞类型列表
    balanced_train_cell_type = [sample.y.item() for sample in balanced_train_graph]

    return balanced_train_graph, balanced_train_cell_type

def preprocess_graph(dataset_atac, graph, test_cell, max_random_sample=500, seed=42):
    dataset_graph = graph.copy()
    train_graph_1, test_graph, train_cell_type_1, test_cell_type = process_samples(dataset_atac, dataset_graph, test_cell)
    torch_vector = train_graph_1[0].id_vec
    train_graph_2, train_cell_type_2 = process_samples_add_random(graph, torch_vector, max_samples=max_random_sample)
    train_graph = train_graph_1 + train_graph_2
    train_cell_type = train_cell_type_1 + train_cell_type_2
    balanced_train_graph, balanced_train_cell_type =  balance_classes(train_graph, train_cell_type, seed=seed)
    return balanced_train_graph, test_graph