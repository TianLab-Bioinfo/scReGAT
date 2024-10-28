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
    temp_ts = torch.tensor(sc_dict[sample.cell])
    sample.sc_vec = temp_ts / temp_ts.sum()
    
    if random.randint(0, 3) == 0:
        test_graph.append(sample)
        test_cell_type.append(sample.y.item())
    else:
        train_graph.append(sample)
        train_cell_type.append(sample.y.item())

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
            nn.Linear(2, 12),
            nn.LayerNorm(12),
            nn.LeakyReLU(),
            nn.Linear(12, edge_embedding_dim)
        )

        # GAT layers with increased heads and self-loops
        self.NN_conv1 = GATConv(node_output_dim, hidden_channels, heads=num_head_1, dropout=dropout, edge_dim=edge_embedding_dim, add_self_loops=False)
        self.NN_flatten1 = nn.Linear(num_head_1 * hidden_channels, hidden_channels)

        self.NN_conv2 = GATConv(hidden_channels, hidden_channels, heads=num_head_2, dropout=dropout, add_self_loops=False)
        self.NN_flatten2 = nn.Linear(num_head_2 * hidden_channels, hidden_channels)

        self.dropout = nn.Dropout(0.2)  
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU()
        
        self.NN_cell_type = nn.Sequential(
            nn.Linear(4457, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 6),
            nn.Softmax()
        )
        

    def forward(self, seq_data, raw_x, edge_index, edge_tf, batch, gene_num):
        data = raw_x
        seq_data = self.NN_seq(seq_data)
        data = torch.cat((data, seq_data), dim=1)
        data = self.NN_node(data)
        
        hidden_edge_input = torch.cat((raw_x[edge_index[0]], raw_x[edge_index[1]]), dim=1)
        hidden_edge = self.NN_edge(hidden_edge_input).sigmoid()
        model.edge = torch.median(hidden_edge, dim=1)[0]
        data, atten_w1 = self.NN_conv1(data, edge_index, edge_attr=hidden_edge, return_attention_weights=True)
        data_1 = self.leaky(self.NN_flatten1(data))

        data_2, atten_w2 = self.NN_conv2(data_1, edge_tf, return_attention_weights=True)
        data_2 = self.leaky(self.NN_flatten2(data_2))
        
        data = data_1 + data_2
        self.data = data
        cell_type = self.NN_cell_type(torch.mean(data[:gene_num], dim=1))
        gene_out = -F.log_softmax(data[:gene_num], dim=1)[:, 0]
        return gene_out, atten_w1, cell_type

import torch
import torch.nn as nn


import torch
import torch.nn as nn

class EdgeSparsityLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=5.0, threshold=0.1):
        """
        自定义稀疏性损失函数
        :param alpha: 控制L1稀疏正则化的权重
        :param beta: 控制高权重边的激励
        :param gamma: 控制阈值惩罚的权重
        :param threshold: 边权重的阈值，小于该值的边将受到额外惩罚
        """
        super(EdgeSparsityLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold

    def forward(self, edge_weights):
        """
        计算自定义稀疏性损失
        :param edge_weights: 模型生成的边权重向量 (torch.Tensor)
        :return: 计算出的稀疏性损失 (torch.Tensor)
        """
        # L1 正则化：鼓励稀疏性
        sparsity_loss = self.alpha * torch.sum(torch.abs(edge_weights))

        # 高权重激励：放大较大的权重
        amplification_loss = -self.beta * torch.sum(torch.log(edge_weights + 1e-6))

        # 阈值惩罚：将小于阈值的边权重推向 0
        threshold_mask = edge_weights < self.threshold
        threshold_penalty = self.gamma * torch.sum(torch.pow(edge_weights[threshold_mask], 2))

        # 总损失：稀疏性、激励和阈值惩罚的组合
        loss = sparsity_loss + amplification_loss + threshold_penalty
        return loss

import torch
import random
from tqdm import tqdm

gene_num = len(sample.y_exp)
device = 'cuda:3'
model.to(device)
loss_exp = torch.nn.KLDivLoss(reduction='batchmean')
# loss_exp = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epoch = 20
drop_edge_rate = 0.1
attention_reg_weight = 0.01
sparse_loss_weight = 0.0001
batch_size = 1
criterion_sparse = EdgeSparsityLoss(alpha=1.0, beta=1.0, gamma=5.0, threshold=0.1)
criterion2 = torch.nn.CrossEntropyLoss()
max_grad_norm = 1.0

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

for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    running_loss1 = 0.0
    running_attention_loss = 0.0  # 记录 attention 正则化损失
    running_sparse_loss = 0.0  # 记录稀疏损失
    random.shuffle(train_graph)
    train_loader = DataLoader(train_graph[:200], batch_size=batch_size, shuffle=True)
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}", unit="batch")
    for idx, sample in enumerate(progress_bar):
        gene_num = sample.y_exp.shape[0]
        optimizer.zero_grad()
        edge_index_dropped = sample.edge_index
        gene_pre, atten, cell_pre = model(
            sample.seq_data.to(device),
            sample.x.to(device),
            sample.edge_index.to(device),
            sample.edge_tf.T.to(device),
            sample.batch.to(device), gene_num
        )
        index = torch.where(sample.id_vec == 1)[0]
        loss1 = -loss_exp(gene_pre.flatten(), sample.y_exp.to(device))
        loss_cell = criterion2(cell_pre.unsqueeze(0), sample.y.to(device))
        attention_variance = torch.var(atten[1], dim=1).mean()
        attention_loss = attention_reg_weight * (2.0 - attention_variance)  # 鼓励方差
        edge_temp = model.edge.flatten()
        loss2 = sparse_loss_weight * criterion_sparse(edge_temp)
        loss = loss1 + attention_loss + loss_cell + loss2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        # 更新运行中的损失
        running_loss += loss.item()
        running_loss1 += loss1.item()
        running_attention_loss += attention_loss.item()  # 记录注意力正则化损失
        running_sparse_loss += loss2.item()  # 记录稀疏损失
        
        # 更新进度条，显示平均损失
        progress_bar.set_postfix(
            loss=running_loss / (progress_bar.n + 1),
            loss1=running_loss1 / (progress_bar.n + 1),
            attention_loss=running_attention_loss / (progress_bar.n + 1),
            sparse_loss=running_sparse_loss / (progress_bar.n + 1)
        )
        
        # 清空 CUDA 缓存，管理显存
        torch.cuda.empty_cache()
    
    print(f"Epoch [{epoch+1}/{num_epoch}], Average Loss: {running_loss / len(train_loader):.4f}, "
          f"Loss1: {running_loss1 / len(train_loader):.4f}, Attention Loss: {running_attention_loss / len(train_loader):.4f}, "
          f"Sparse Loss: {running_sparse_loss / len(train_loader):.4f}")

test_loader = DataLoader(test_graph, batch_size=1, shuffle=True, pin_memory=True)
model.train()
test_loss = 0.0 

device = 'cuda:0'
model.to(device)

cell_type = []
test_barcodes = []
with torch.no_grad():
    cell_link_atten = []
    cell_link_activity = []
    cell_link_edge = []
    cell_link_enhancer = []
    for idx, sample in tqdm(enumerate(test_loader), total=len(test_loader), desc="Processing Batches"):
        gene_pre, atten, _ = model(sample.seq_data.to(device),
                                sample.x.to(device), 
                                sample.edge_index.to(device), 
                                sample.edge_tf.T.to(device), 
                                sample.batch.to(device), gene_num)
        test_barcodes.append(sample.cell)
        cell_type.append(sample.y.item())
        import torch

        # 对 model.edge.flatten() 进行 0-1 缩放
        edge_flattened = model.edge.flatten().to('cpu')
        min_val = torch.min(edge_flattened)
        max_val = torch.max(edge_flattened)
        normalized_edge = (edge_flattened - min_val) / (max_val - min_val + 1e-10)  # 防止除以 0

        # 将归一化后的张量添加到 cell_link_edge
        cell_link_edge.append(normalized_edge)


        flattened_sample_x = sample.to(device).x[atten[0][1].cpu()].flatten()
        flattened_sample_enhancer = sample.to(device).x[atten[0][0].cpu()].flatten()
        atten1_max = torch.mean(atten[1], dim=1).to('cpu')
        cell_link_activity.append(flattened_sample_x.to('cpu'))
        cell_link_atten.append(atten1_max)
        cell_link_enhancer.append(flattened_sample_enhancer.to('cpu'))
#         edges = model.edge.flatten().cpu().numpy()
        torch.cuda.empty_cache()
