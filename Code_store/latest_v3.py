import os
import pandas as pd
import torch
import anndata as ad
from torch_geometric.loader import DataLoader
import scanpy as sc
from scregat.data_process import prepare_model_input,sum_counts,plot_edge, ATACGraphDataset
import numpy as np
from scregat.model import train_scregat, explain_model_ig
import pickle

import joblib
with open('./dataset_atac.pkl', 'rb') as f:
    dataset_atac = joblib.load(f)

    
dataset_graph = ATACGraphDataset('./input_graph/')
test_cell = pd.read_csv('./test_cell_name.txt')['x'].values


#import os
import pandas as pd
import torch
import anndata as ad
from torch_geometric.loader import DataLoader
import scanpy as sc
from scregat.data_process import prepare_model_input,sum_counts,plot_edge, ATACGraphDataset
import numpy as np
import pickle
import random
from tqdm import tqdm

seq_vec = torch.load('./seq.pth')

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

    seq_data = seq_vec
    sample.seq_data = seq_data
    sample.id_vec = torch_vector

    
    if sample.cell in test_cell:
        test_graph.append(sample)
        test_cell_type.append(sample.y.item())
    else:
        train_graph.append(sample)
        train_cell_type.append(sample.y.item())

import torch
import torch.nn as nn

class EdgeDiversityLoss1(nn.Module):
    def __init__(self, diversity_weight=1.0):
        """
        自定义多样性损失函数
        :param diversity_weight: 控制熵惩罚的权重
        """
        super(EdgeDiversityLoss1, self).__init__()
        self.diversity_weight = diversity_weight

    def forward(self, edge_weights):
        prob_distribution = torch.softmax(edge_weights, dim=0)
        entropy_loss = -self.diversity_weight * torch.sum(prob_distribution * torch.log(prob_distribution + 1e-6), dim=0)
        
        # 添加均匀性损失
        target_distribution = torch.full_like(prob_distribution, 1.0 / edge_weights.numel())
        uniformity_loss = torch.sum((prob_distribution - target_distribution) ** 2)

        # 总损失
        loss = torch.sum(entropy_loss) + uniformity_loss
        return loss

import torch
import torch.nn as nn

class EdgeDiversityLoss2(nn.Module):
    def __init__(self, non_zero_penalty_weight=1.0):
        """
        自定义多样性损失函数
        :param non_zero_penalty_weight: 控制非零惩罚的权重
        """
        super(EdgeDiversityLoss2, self).__init__()
        self.non_zero_penalty_weight = non_zero_penalty_weight

    def forward(self, edge_weights):
        """
        计算自定义多样性损失
        :param edge_weights: 模型生成的边权重矩阵 (torch.Tensor)，形状为 (batch_size, num_edges)
        :return: 计算出的多样性损失 (torch.Tensor)
        """
        # 筛选出非零元素
        non_zero_weights = edge_weights[edge_weights != 0]

        # 计算非零元素的方差并取负
        variance_loss = -torch.var(non_zero_weights)

        # 计算非零惩罚：惩罚零权重的数量
        non_zero_penalty = self.non_zero_penalty_weight * torch.sum((edge_weights == 0).float())

        # 总损失：使用负方差作为多样性损失和非零惩罚的组合
        loss = variance_loss + non_zero_penalty
        return loss

def drop_edges(edge_index, drop_rate=0.1):
    """随机丢弃边，依据设定的丢弃比例。
    Args:
        edge_index (torch.Tensor): 边的张量，形状为 (2, num_edges)。
        drop_rate (float): 要丢弃的边的比例。
    Returns:
        torch.Tensor: 丢弃部分边后的 edge_index。
    """
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) > drop_rate
    return edge_index[:, mask]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class SCReGAT(torch.nn.Module):
    def __init__(self,
                 node_input_dim=2,
                 node_output_dim=8,
                 edge_embedding_dim=8,
                 hidden_channels=16,  # 增加 hidden_channels
                 gat_input_channels=8,
                 gat_hidden_channels=8,  # 增加 GAT 隐藏通道
                 seq_dim=768,
                 seq2node_dim=1,
                 max_tokens=1024,
                 dropout=0.4,  # 增加 dropout
                 num_head_1=16,  # 增加 GAT 头数
                 num_head_2=16):  # 增加 GAT 头数
        super(SCReGAT, self).__init__()

        # Sequence transformation layer (currently commented out in forward)
        self.NN_seq = nn.Sequential(
            nn.Linear(seq_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, seq2node_dim),
        )

        # Node feature transformation with BatchNorm
        self.NN_node = nn.Sequential(
            nn.Linear(node_input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, node_output_dim),
        )

        # Edge feature transformation with BatchNorm
        self.NN_edge = nn.Sequential(
            nn.Linear(3, 16),
            nn.LayerNorm(16),
            nn.LeakyReLU(),
            nn.Linear(16, edge_embedding_dim),
            nn.LeakyReLU()
        )

        # GAT layers with increased heads and self-loops
        self.NN_conv1 = GATConv(node_output_dim, hidden_channels, heads=num_head_1, dropout=dropout, edge_dim=edge_embedding_dim, add_self_loops=False)
        self.NN_flatten1 = nn.Linear(num_head_1 * hidden_channels, hidden_channels)

        self.NN_conv2 = GATConv(hidden_channels, hidden_channels, heads=num_head_2, dropout=dropout, add_self_loops=False)
        self.NN_flatten2 = nn.Linear(num_head_2 * hidden_channels, hidden_channels)

        self.dropout = nn.Dropout(0.1)  
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU()
        
        self.NN_cell_type = nn.Sequential(
          # 手动写上你的gene node
            nn.Linear(, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
          # 手动写上你的gene node
            nn.Linear(64, 9),
            nn.Softmax()
        )
        

    def forward(self, seq_data, raw_x, edge_index, edge_tf, batch, gene_num, gene_id_vec, is_test=False):
        data = raw_x
        seq_data = self.NN_seq(seq_data)
        data = torch.cat((data, seq_data), dim=1)
        data = self.NN_node(data)
         
        hidden_edge_input = torch.cat((raw_x[edge_index[0]] * raw_x[edge_index[1]], raw_x[edge_index[0]], raw_x[edge_index[1]]), dim=1)
        hidden_edge = self.NN_edge(hidden_edge_input).sigmoid()
        
        model.edge = torch.median(hidden_edge, dim=1)[0]
        data, atten_w1 = self.NN_conv1(data, edge_index, edge_attr=hidden_edge, return_attention_weights=True)
        data_1 = self.leaky(self.NN_flatten1(data))

        data_2, atten_w2 = self.NN_conv2(data_1, edge_tf, return_attention_weights=True)
        data_2 = self.leaky(self.NN_flatten2(data_2))
        
        data = data_1 + data_2
        self.data = data
        if is_test == False:
            cell_type = self.NN_cell_type(torch.mean(data[gene_id_vec==1], dim=1))
        else:
            cell_type = None
            pass
        gene_out = -F.log_softmax(data[gene_id_vec==1], dim=1)[:, 0]

        return gene_out, atten_w1, cell_type

def get_split(mt, batch_size):
    edge_flattened = mt
    if edge_flattened.size(0) % batch_size == 0:
        split_edges = edge_flattened.split(edge_flattened.size(0) // batch_size)
    else:
        remainder = edge_flattened.size(0) % batch_size
        edge_flattened = edge_flattened[:-remainder]  # Drop the remainder
        split_edges = edge_flattened.split(edge_flattened.size(0) // batch_size)
    return torch.vstack(split_edges)


model = SCReGAT()
import torch
from tqdm import tqdm

batch_size = 4
test_loader = DataLoader(test_graph, batch_size=batch_size, shuffle=True, pin_memory=True)
model.eval()
device = 'cuda:3'
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
        gene_pre, atten, _ = model(
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
        test_barcodes.extend(sample.cell)  # Ensure on CPU
        cell_type.extend(sample.y.cpu().numpy())

        # Edge processing
        edge_temp = model.edge.flatten().to('cpu')
        edge_lists = edge_temp.split(edge_temp.size(0) // sample_size)

        for edge in edge_lists:
            min_val = edge.min()
            max_val = edge.max()
            normalized_edge = (edge - min_val) / (max_val - min_val + 1e-10)  # Prevent division by zero
            cell_link_edge.append(normalized_edge)

        # Flatten and process attention data
        atten_x_indices = atten[0][1].cpu()
        atten_enhancer_indices = atten[0][0].cpu()
        flattened_sample_x = sample.x[atten_x_indices].flatten().to('cpu')
        flattened_sample_enhancer = sample.x[atten_enhancer_indices].flatten().to('cpu')
        atten1_max = atten[1].mean(dim=1).to('cpu')

        # Split and collect
        for x, enhancer, atten_ in zip(
            flattened_sample_x.split(flattened_sample_x.size(0) // sample_size),
            flattened_sample_enhancer.split(flattened_sample_enhancer.size(0) // sample_size),
            atten1_max.split(atten1_max.size(0) // sample_size)
        ):
            cell_link_activity.append(x)
            cell_link_atten.append(atten_)
            cell_link_enhancer.append(enhancer)

        # Clear cache
        torch.cuda.empty_cache()

# Optionally return or further process the collected data
cell_link_atten = torch.stack(cell_link_atten)
cell_link_edge = torch.stack(cell_link_edge)
cell_link_activity = torch.stack(cell_link_activity)
cell_link_enhancer = torch.stack(cell_link_enhancer)
