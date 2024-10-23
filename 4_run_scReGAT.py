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

with open('./dataset_atac.pkl', 'rb') as f:
    dataset_atac = pickle.load(f)

dataset_graph = ATACGraphDataset('./input_graph/')
seq_vec = torch.load('./seq.pth')

# 你要研究的barcode
test_cell = pd.read_csv('./test_cell.txt', header=None)

# 如果没指定，呃，那你可以随机切分
cells = dataset_graph.cell
random.shuffle(cells)
split_index = int(len(cells) * 0.6)
train_cell = cells[:split_index]
test_cell = cells[split_index:]

# 然后我们打上基因的启动子peak节点标记
data = dataset_atac.array_peak
torch_vector = torch.zeros(len(data))
for idx, item in enumerate(data):
    if item.startswith('chr'):
        torch_vector[idx] = 0 
    else:
        torch_vector[idx] = 1 
print(torch_vector)




# 现在我们可以生成修改原始图
train_graph = []
test_graph = []

for i, sample in tqdm(enumerate(dataset_graph), total=len(dataset_graph), desc='Processing samples'):
# 此时的sample代表一个细胞的图
    seq_data = seq_vec
    sample.seq_data = seq_data
    sample.id_vec = torch_vector
  
    # 请注意，如果你要添加边
    # 可以直接修改 sample.edge_index
    # sample.edge_index[1]是promoter节点的索引
    # sample.edge_index[0]是promoter节点对应的peak节点的索引
    # 每个索引代表的peak或promoter节点的含义可以通过dataset.array_peak查询
    # 比如
    # dataset_atac.array_peak
    # array(['ACAP3', 'VWA1', 'PRDM16', ..., 'chrX-154762391-154763458',
    # 
    # sample.edge_index
    # tensor([[21068, 21083, 21084,  ..., 56014, 56041, 56043],
    #     [    0,     1,     1,  ...,  1528,  1529,  1529]], device='cuda:3')
    # 0就代表了dataset_atac.array_peak[0]: 'ACAP3'
    # 21068就代表了dataset_atac.array_peak[21068] : 'chr1-1307926-1308733'
    # 
    # 假设我们要添加基因'ACAP3'和'chr1-1307926-1308733'，可以：
    # sample.edge_index[0] = torch.cat((sample.edge_index[0], torch.tensor([21068])))
    # sample.edge_index[1] = torch.cat((sample.edge_index[1], torch.tensor([0])))
    # 你当然也可以直接删除边，scReGAT采用的架构可以无视图的点边结构


    # 当然我们也可以在这里直接修改细胞的target基因表达y_exp
    # sample.y_exp = torch.tensor([新的值array])就可以了
    # 意味着我们可以轻松改成单细胞表达

    if sample.cell in test_cell:
        test_graph.append(sample)
    else:
        train_graph.append(sample)



# 下面是模型，这是KL散度版本
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class SCReGAT(torch.nn.Module):
    def __init__(self,
                 node_input_dim=2,
                 node_output_dim=8,
                 edge_embedding_dim=8,
                 hidden_channels=8,
                 gat_input_channels=8,
                 gat_hidden_channels=8,
                 seq_dim=768,
                 seq2node_dim=1,
                 max_tokens=1024,
                 dropout=0.2,
                 num_head_1=8,
                 num_head_2=8):
        super(SCReGAT, self).__init__()
        torch.manual_seed(12345)

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
            nn.LayerNorm(128),  # BatchNorm added after Linear
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),  # BatchNorm added
            nn.LeakyReLU(),
            nn.Linear(64, node_output_dim),
        )

        # Edge feature transformation with BatchNorm
        self.NN_edge = nn.Sequential(
            nn.Linear(2, 12),
            nn.LayerNorm(12),  # BatchNorm added
            nn.LeakyReLU(),
            nn.Linear(12, edge_embedding_dim)
        )

        # GAT layers with dropout
        self.NN_conv1 = GATConv(node_output_dim, hidden_channels, heads=num_head_1, dropout=dropout, edge_dim=edge_embedding_dim, add_self_loops=False)
        self.NN_flatten1 = nn.Linear(num_head_1 * hidden_channels, hidden_channels)

        self.NN_conv2 = GATConv(hidden_channels, hidden_channels, heads=num_head_2, dropout=dropout, add_self_loops=False)
        self.NN_flatten2 = nn.Linear(num_head_2 * hidden_channels, hidden_channels)

        # Adapters and dropout layers
        self.NN_adapter_barcode = nn.Linear(hidden_channels, 1)
        self.NN_adapter_bulk = nn.Linear(hidden_channels, 1, bias=True)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU()


    def forward(self, seq_data, raw_x, edge_index, edge_tf, batch, gene_num, tag):
        data = raw_x
        seq_data = self.NN_seq(seq_data)
        data = torch.cat((data, seq_data), dim=1)
        data = self.NN_node(data)
        
        hidden_edge_input = torch.cat((raw_x[edge_index[0]], raw_x[edge_index[1]]), dim=1)
        hidden_edge = self.NN_edge(hidden_edge_input).sigmoid()
    
        data, atten_w1 = self.NN_conv1(data, edge_index, edge_attr=hidden_edge, return_attention_weights=True)
        data_1 = self.leaky(self.NN_flatten1(data))

        data_2, atten_w2 = self.NN_conv2(data_1, edge_tf, return_attention_weights=True)
        data_2 = self.leaky(self.NN_flatten2(data_2))
        
        data = data_1 + data_2
        
        # data = self.NN_adapter_bulk(data)
        self.data = data
        gene_out = -F.log_softmax(data, dim=1)[:, 0]
        return gene_out, atten_w1


# 你懂的
model = SCReGAT()
train_loader = DataLoader(train_graph, batch_size=1, shuffle=True)
gene_num = len(sample.y_exp)


# 训练部分

# 假设你已经定义了 model, train_graph 等
device = 'cuda:3'
model.to(device)

# 使用 KL 散度损失
loss_exp = torch.nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epoch = 20
drop_edge_rate = 0.1

for epoch in range(num_epoch):
    model.train()  
    
# 以下是随机抽样训练，你也可以用
#     # 随机跳过部分 epoch
#     if random.randint(1, 4) != 2:
#         continue
#     else:
#         random.shuffle(train_graph)
#         train_loader = DataLoader(train_graph[:500], batch_size=1, shuffle=True)

    running_loss = 0.0  
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}", unit="batch")  
    count = 0

    for sample in progress_bar:
        gene_num = sample.y_exp.shape[0]
        optimizer.zero_grad()  

        # 执行前向传播
        gene_pre, atten = model(
            sample.seq_data.to(device),
            sample.x.to(device), 
            sample.edge_index.to(device),
            sample.edge_tf.T.to(device), 
            sample.batch.to(device), 
            gene_num, 
            sample.id_vec.to(device)
        )

        # 我们只考虑那些活性 > 0的promoter对应的基因表达，因为这样可以避免将0纳入
        index = torch.where(sample.x[:gene_num] > 0)[0]
        gene_pre_log_prob = torch.log_softmax(gene_pre[index].flatten(), dim=-1)  
        loss = -loss_exp(gene_pre[index].flatten(), sample.y_exp[index].to(device)) 
       
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))  # 更新进度条的损失值
       
    print(f"Epoch [{epoch+1}/{num_epoch}], Average Loss: {running_loss / len(train_loader):.4f}")


# 测试
test_loader = DataLoader(test_graph, batch_size=1, shuffle=False, pin_memory=True)
model.eval()  # 切换到评估模式
test_loss = 0.0  # 初始化测试集的总损失

# 使用 torch.no_grad() 禁用梯度计算
device = 'cuda:3'
cell_type = []
with torch.no_grad():
    cell_link_mt = []
    for idx, sample in tqdm(enumerate(test_loader), total=len(test_loader), desc="Processing Batches"):
        gene_pre, atten = model(sample.seq_data.to(device),
                                  sample.x.to(device), 
                                  sample.edge_index.to(device), 
                                  sample.edge_tf.T.to(device), 
                                  sample.batch.to(device), gene_num, sample.id_vec.to(device))
        cell_type.append(sample.y.item())

        temp = torch.mean(atten[1], dim=1) * sample.to(device).x[atten[0][1].cpu()].flatten()
        cell_link_mt.append(temp)

# cell_link_mt 这个矩阵是[n_cell * n_edge]
# 其中边的与sample.edge_index的索引完全一致
# cell_type是你用于测试用的细胞类别
