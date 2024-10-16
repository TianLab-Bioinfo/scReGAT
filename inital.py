# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @file: model.py
# @time: 2023/2/1 14:30


from time import time
import os
import subprocess
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, device
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import captum.attr as attr
from audtorch.metrics.functional import pearsonr
from torch_sparse import SparseTensor, set_diag, matmul
from scipy import stats
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from scregat.data_process import ATACDataset, ATACGraphDataset


class GATConvW(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            edge_dim: Optional[int] = None,
            fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True,
            share_weights: bool = False,
            **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        _alpha: OptTensor
        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, edge_weight: OptTensor = None,
                return_attention_weights: bool = None):
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_weight=edge_weight,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor, edge_weight: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x += edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        x_j = x_j * alpha.unsqueeze(-1)

        if edge_weight is None:
            return x_j
        else:
            edge_weight = edge_weight.view(-1, 1, 1)
            return edge_weight * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class SCReGAT(torch.nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, num_head: int, num_gene: int,
                 num_celltype: int, num_nodes: int):
        super(SCReGAT, self).__init__()
        torch.manual_seed(12345)
        self.num_nodes = num_nodes
        self.num_gene = num_gene

        self.lin1_x = nn.Linear(input_channels, hidden_channels)
        self.lin1_edge = nn.Linear(input_channels, hidden_channels)
        self.conv1 = GATConvW(hidden_channels, hidden_channels, heads=num_head, dropout=0.5,
                              edge_dim=hidden_channels, add_self_loops=False)
        self.ln_1 = LayerNorm(self.num_nodes)
        self.lin2 = nn.Linear(1, hidden_channels)

        self.conv2 = GATConvW(hidden_channels, hidden_channels,
                              heads=1, dropout=0.5, add_self_loops=False)
        self.ln_2 = LayerNorm(num_gene)

        self.lin3 = nn.Linear(num_gene, num_gene)
        self.lin4 = nn.Linear(num_gene, num_celltype)

    def forward(self, x: Tensor, edge_index: Tensor, edge_tf: Tensor, batch: Tensor,
                edge_weight: Optional[Tensor] = None, edge_weight_tf: Optional[Tensor] = None):
        batchsize = len(torch.unique(batch))
        self.raw_x = x
        self.batch = batch
        x_edge = x[edge_index[0, :], :] * x[edge_index[1, :], :]
        self.x_edge_index = edge_index
        self.x_edge_raw = x_edge
        
        x = self.lin1_x(x).sigmoid()
        self.lin1_x_x = x
        
        x_edge = self.lin1_edge(x_edge).sigmoid()
        self.lin1_edge_x_edge = x_edge
        self.x_edge = x_edge
        x, atten_w = self.conv1(x, edge_index, x_edge, edge_weight, return_attention_weights=True)
        self.conv1_x = x
        x = x.view(batchsize, self.num_nodes, -1)
        x = torch.mean(x, dim=-1, keepdim=False)
        # x_1 = self.ln_1(x)
        x_1 = x
        x = x_1.unsqueeze(-1).view(batchsize*self.num_nodes, -1)
        x = self.lin2(x).sigmoid()

        x, atten_w2 = self.conv2(x, edge_tf, edge_attr=None, edge_weight=edge_weight_tf,
                                 return_attention_weights=True)
        self.conv2_x = x
        x = x.view(batchsize, self.num_nodes, -1)
        x = torch.mean(x, dim=-1, keepdim=False)
        x_2 = x_1 + x
        self.x_all_1 = x
        self.x_all_2 = x_1
        self.x_all = x_2
        x_gene = x_2[:, :self.num_gene]
        return '', F.log_softmax(x_gene, dim=1), atten_w, atten_w2, x_edge


class MyLoss(nn.Module):
    def __init__(self, lambda_1: float, weight_label: Tensor):
        super(MyLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.weight_label = weight_label
        self.loss_exp = torch.nn.KLDivLoss(log_target=False, reduction='batchmean')

    def forward(self, out_exp: Tensor, true_exp: Tensor, x_edge: Tensor):
        exp_loss = self.loss_exp(out_exp, true_exp)
        return exp_loss


def train(model: Module, criterion: Module, optimizer: Optimizer, 
          use_device: device, loader: DataLoader):
    model.train()
    list_loss = []
    for data in loader:  # Iterate in batches over the training dataset.
        data = data.to(use_device)
        edge_tf_input = data.edge_tf.T
        batch_tf = []
        batchsize = len(torch.unique(data.batch))
        num_peaks = data.x.shape[0] // batchsize
        num_tfpair = edge_tf_input.shape[1] // batchsize
        for idx_tf in range(batchsize):
            batch_tf.extend([idx_tf * num_peaks for _ in range(num_tfpair)])
        tensor_batch_tf = torch.tensor([batch_tf, batch_tf]).to(use_device)
        edge_tf_input = edge_tf_input + tensor_batch_tf
        _, out2, out_atten, out_atten2, out_edge = \
            model(data.x, data.edge_index, edge_tf_input, data.batch)
        loss = criterion(out2, data.y_exp.view(out2.shape), out_edge)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        list_loss.append(loss.cpu().detach().numpy())
    loss_cat = np.array(list_loss)
    return np.mean(loss_cat)


def test(model: Module, use_device: device, loader: DataLoader):
    with torch.no_grad():
        correct = 0
        list_corr = []
        for data in loader:  # Iterate in batches over the training/test dataset.c
            data = data.to(use_device)
            edge_tf_input = data.edge_tf.T
            batchsize = len(torch.unique(data.batch))
            num_peaks = data.x.shape[0] // batchsize
            batch_tf = []
            num_tfpair = edge_tf_input.shape[1] // batchsize
            for idx_tf in range(batchsize):
                batch_tf.extend([idx_tf * num_peaks for _ in range(num_tfpair)])
            tensor_batch_tf = torch.tensor([batch_tf, batch_tf]).to(use_device)
            edge_tf_input = edge_tf_input + tensor_batch_tf
            _, out2, out_atten, out_atten2, edge = \
                model(data.x, data.edge_index, edge_tf_input, data.batch)
           
            list_corr.append(pearsonr(
                torch.exp(out2.cpu()), torch.exp(data.y_exp.view(out2.shape).cpu())))
        corr_cat = torch.cat(list_corr, dim=0)

    return torch.median(corr_cat)


def predict(model: Module, use_device: device, loader: DataLoader):
    with torch.no_grad():
        list_exp = []
        list_y = []
        list_y_exp = []
        list_cell = []
        list_pred = []
        for data in loader:
            data = data.to(use_device)
            _, out2, out_atten, out_atten2. edge = \
                model(data.x, data.edge_index, data.edge_tf.T, data.batch)  
            list_exp.append(out2.cpu().detach().numpy())
          
            list_y_exp.append(data.y_exp.view(out2.shape).cpu().detach().numpy())
            

    return np.concatenate(list_exp), np.concatenate(list_y_exp)


def train_scregat(path_data_root: str, dataset_atac: ATACDataset, dir_model: str,
                  use_device: device, hidwidth: int = 16, numhead: int = 8,
                  learning_rate: float = 1e-3, num_epoch: int = 20,
                  split_prop: float = 0.6, batch_size: int = 16, print_process: bool = False):
    # read data
    # file_atac_test = os.path.join(path_data_root, 'dataset_atac.pkl')
    # with open(file_atac_test, 'rb') as r_pkl:
    #     dataset_atac = pickle.loads(r_pkl.read())
    path_graph_input = os.path.join(path_data_root, 'input_graph')
    dataset_atac_graph = ATACGraphDataset(path_graph_input)

    torch.manual_seed(12345)
    dataset = dataset_atac_graph.shuffle()
    # split_prop = 0.8
    num_split = int(len(dataset) * split_prop)
    train_dataset = dataset[:num_split]
    test_dataset = dataset[num_split:]

    # batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # use_device = torch.use_device("cuda:0" if torch.cuda.is_available() else "cpu")
    peaks = dataset_atac.array_peak
    mask_numpy = np.array([0 if peak[:3] == 'chr' else 1 for peak in peaks])
    number_gene = int(np.sum(mask_numpy))

    list_weights = []
    for i in range(len(dataset_atac.array_celltype)):
        sub_dataset = [data for data in train_dataset if data.y == i]
        sub_len = len(sub_dataset)
        sub_weight = len(train_dataset)/sub_len
        list_weights.append(sub_weight)
    criterion = MyLoss(1, torch.tensor(list_weights).to(use_device))

    # hidwidth, numhead = 16, 8
    model = SCReGAT(input_channels=dataset.num_node_features,
                    hidden_channels=hidwidth, num_head=numhead,
                    num_gene=number_gene, num_celltype=dataset.num_classes,
                    num_nodes=dataset_atac_graph[0].num_nodes).to(use_device)

    # train scReGAT
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    list_loss = []
    list_train_acc = []
    list_train_corr = []
    list_test_acc = []
    list_test_corr = []
    for epoch in range(num_epoch):
        loss_t = train(model, criterion, optimizer, use_device, train_loader)
        train_acc, train_corr = test(model, use_device, train_loader)
        test_acc, test_corr = test(model, use_device, test_loader)
        
        list_loss.append(loss_t)
        list_train_acc.append(train_acc)
        list_train_corr.append(train_corr)
        list_test_acc.append(test_acc)
        list_test_corr.append(test_corr)
        if print_process:
            print(f'Epoch: {epoch:03d}, Total loss: {loss_t:.4f}, Label loss: {loss_1:.4f}, '
                  f'Expression loss: {loss_2:.4f} \n'
                  f'Train Acc: {train_acc:.4f}, Train Corr: {train_corr:.4f}, \n'
                  f'Test Acc: {test_acc:.4f}, Test Corr: {test_corr:.4f}')
            print(f'L1 Loss: {loss_3:.4f}')
    # dir_model = 'scReGAT'
    path_model = os.path.join(path_data_root, dir_model)
    if not os.path.exists(path_model):
        os.mkdir(path_model)

    # save results
    df_res = pd.DataFrame({"loss": list_loss,
                           'train_acc': list_train_acc, 'test_acc': list_test_acc,
                           'train_corr': list_train_corr, 'test_corr': list_test_corr})
    file_res = \
        os.path.join(path_model,
                     f'res_batch_size_{batch_size}_hidwidth_{hidwidth}_numhead_{numhead}_'
                     f'lr_{learning_rate}_numepoch_{num_epoch}_split_{split_prop}.txt')
    df_res.to_csv(file_res, sep='\t')

    # save scReGAT
    file_atac_model = \
        os.path.join(path_model,
                     f'Model_batch_size_{batch_size}_hidwidth_{hidwidth}_numhead_{numhead}_'
                     f'lr_{learning_rate}_numepoch_{num_epoch}_split_{split_prop}.pt')
    torch.save(model, file_atac_model)

    return model, test_dataset


def model_forward(edge_mask: Tensor, data: Data, model: Module, use_device: device, gene_idx: int):
    edge_tf_input = data.edge_tf.T
    batchsize = len(torch.unique(data.batch))
    num_peaks = data.x.shape[0] // batchsize
    batch_tf = []
    num_tfpair = edge_tf_input.shape[1] // batchsize
    for idx_tf in range(batchsize):
        batch_tf.extend([idx_tf * num_peaks for _ in range(num_tfpair)])
    tensor_batch_tf = torch.tensor([batch_tf, batch_tf]).to(use_device)
    edge_tf_input = edge_tf_input + tensor_batch_tf
    _, out_exp, _, _, _  = model(data.x, data.edge_index, edge_tf_input, data.batch, edge_mask)
    return out_exp[:, gene_idx]


def explain_model_ig(dataset_atac: ATACDataset, model: Module, use_device: device,
                     explain_dataset: ATACGraphDataset, file_weight: str,
                     method: str = 'ixg', batch_size: int = 32, print_process: bool = False):
    test_loader_explain = DataLoader(explain_dataset, batch_size=batch_size, shuffle=False)
    if method == 'ixg':
        ixg = attr.InputXGradient(model_forward)
    elif method == 'ig':
        ig = attr.IntegratedGradients(model_forward)
    else:
        print('InputError')
        return
    peaks = dataset_atac.array_peak
    idx_edge = explain_dataset[0].edge_index.numpy().astype('int32')
    list_array = []
    list_pairs = []
    # idx_gene = 305
    for idx_gene in range(dataset_atac.df_rna.shape[1]):
        list_edge_array = []
        gene = peaks[idx_gene]
        for data in test_loader_explain:
            data = data.to(use_device)
            # x_input = data.x
            # target = data.y
            # out_c, out_e, _ = scReGAT(data.x, data.edge_index, data.batch)
            # pred = out_c.argmax(dim=1)
            input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(use_device)
            if method == 'ixg':
                mask = ixg.attribute(
                    input_mask, additional_forward_args=(data, model, use_device, idx_gene))
            else:
                mask = ig.attribute(
                    input_mask, n_steps=10,
                    additional_forward_args=(data, model, use_device, idx_gene))
            # mask = ig.attribute(
            #     input_mask, n_steps=10,
            #     additional_forward_args=(data, scReGAT, idx_gene),
            #     internal_batch_size=data.edge_index.shape[1])
            batch_size = len(torch.unique(data.batch))
            num_col = mask.shape[0]//batch_size
            mask = mask.view(batch_size, num_col)
            edge_mask = mask.cpu().detach().numpy().astype('float32')
            # edge_mask = np.abs(mask.cpu().detach().numpy())
            # edge_mask = edge_mask / np.max(edge_mask, axis=1)[:, np.newaxis]
            list_edge_array.append(edge_mask)
        weight_array = np.concatenate(list_edge_array, axis=0)
        idx_peak = idx_edge[0, idx_edge[1, :] == idx_gene]
        sub_pair = [(gene, peaks[idx]) for idx in idx_peak]
        sub_array = weight_array[:, idx_edge[1, :] == idx_gene]
        list_array.append(sub_array)
        list_pairs.extend(sub_pair)
        if print_process:
            if idx_gene % 50 == 0:
                print(f"Calculating progress: {idx_gene+1} genes completed")

    array_merge = np.concatenate(list_array, axis=1)
    list_cell = [item for data in test_loader_explain for item in data.cell]
    df_merge = pd.DataFrame(array_merge, index=list_cell, columns=list_pairs)

    # save weight
    # file_weight = os.path.join(path_weight, 'weight_ixg_gene_1000.tsv')
    df_merge.to_csv(file_weight, sep='\t')

    return df_merge


if __name__ == '__main__':
    time_start = time()

    time_end = time()
    print(time_end - time_start)
