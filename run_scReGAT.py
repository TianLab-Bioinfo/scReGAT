import os
import numpy as np
import pandas as pd
import pickle
import random
from time import time
from collections import defaultdict
from typing import Optional, Mapping, List, Union
import pybedtools
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
        Custom diversity loss using entropy and uniformity.

        :param diversity_weight: Weight controlling the entropy penalty.
        """
        super(EdgeDiversityLoss1, self).__init__()
        self.diversity_weight = diversity_weight

    def forward(self, edge_weights):
        """
        Forward computation of the diversity loss.

        :param edge_weights: Tensor of edge weights (any shape).
        :return: Scalar tensor representing the diversity loss.
        """
        # Compute softmax over edge weights to get probability distribution
        prob_distribution = torch.softmax(edge_weights, dim=0)

        # Entropy loss: encourage spread-out distribution (maximize entropy)
        entropy_loss = -self.diversity_weight * torch.sum(
            prob_distribution * torch.log(prob_distribution + 1e-6), dim=0
        )

        # Uniformity loss: penalize deviation from uniform distribution
        target_distribution = torch.full_like(prob_distribution, 1.0 / edge_weights.numel())
        uniformity_loss = torch.mean((prob_distribution - target_distribution) ** 2)

        # Total loss = entropy + uniformity
        loss = torch.mean(entropy_loss) + uniformity_loss
        return loss


class EdgeDiversityLoss2(nn.Module):
    def __init__(self, non_zero_penalty_weight=1.0):
        """
        Custom diversity loss based on variance and sparsity.

        :param non_zero_penalty_weight: Weight controlling penalty for zero edges.
        """
        super(EdgeDiversityLoss2, self).__init__()
        self.non_zero_penalty_weight = non_zero_penalty_weight

    def forward(self, edge_weights):
        """
        Forward computation of diversity loss.

        :param edge_weights: Tensor of edge weights (e.g., shape [batch_size, num_edges]).
        :return: Scalar tensor representing the diversity loss.
        """
        # Select non-zero weights only
        non_zero_weights = edge_weights[edge_weights != 0]

        # Variance loss: encourage diverse (spread out) non-zero edge weights
        variance_loss = -torch.var(non_zero_weights)

        # Sparsity penalty: penalize too many zero weights
        non_zero_penalty = self.non_zero_penalty_weight * torch.mean((edge_weights == 0).float())

        # Total loss = -variance + zero-weight penalty
        loss = variance_loss + non_zero_penalty
        return loss



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class SCReGAT(torch.nn.Module):
    def __init__(self,
                 node_input_dim=1,
                 node_output_dim=8,
                 edge_embedding_dim=8,
                 hidden_channels=16,
                 gat_input_channels=8,
                 gat_hidden_channels=8,
                 seq_dim=768,
                 seq2node_dim=1,
                 max_tokens=1024,
                 dropout=0.1,
                 num_head_1=16,
                 num_head_2=16):
        super(SCReGAT, self).__init__()

        # Sequence-level embedding network
        self.NN_seq = nn.Sequential(
            nn.Linear(seq_dim, 512),
            nn.BatchNorm1d(512),  # Normalize for stable training
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, seq2node_dim)
        )

        # Node-level transformation
        self.NN_node = nn.Sequential(
            nn.Linear(node_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, node_output_dim)
        )

        # Edge embedding network
        self.NN_edge = nn.Sequential(
            nn.Linear(3, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(16, edge_embedding_dim),
            nn.BatchNorm1d(edge_embedding_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

        # Residual network applied after GAT layers
        self.NN_res = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 16)
        )

        # First GAT layer: multi-head attention with edge features
        self.NN_conv1 = GATConv(
            node_output_dim, hidden_channels, heads=num_head_1,
            dropout=dropout, edge_dim=edge_embedding_dim, add_self_loops=False
        )
        self.NN_flatten1 = nn.Linear(num_head_1 * hidden_channels, hidden_channels)

        # Second GAT layer (e.g., for transformed edges)
        self.NN_conv2 = GATConv(
            hidden_channels, hidden_channels, heads=num_head_2,
            dropout=dropout, add_self_loops=False
        )
        self.NN_flatten2 = nn.Linear(num_head_2 * hidden_channels, hidden_channels)

        # Dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, seq_data, raw_x, edge_index, edge_tf, batch, gene_num, gene_id_vec, is_test=False):
        """
        Forward pass of SCReGAT.

        :param seq_data: Input sequence features (not used in current logic)
        :param raw_x: Raw node features
        :param edge_index: Graph connectivity for first GAT layer
        :param edge_tf: Graph connectivity for second GAT layer
        :param batch: Batch vector for graph batching
        :param gene_num: Total number of genes (unused here)
        :param gene_id_vec: Binary vector indicating gene positions
        :param is_test: Flag to toggle test-time behavior (optional)
        :return: Log-softmax loss values for selected genes and attention weights from GAT layer 1
        """
        p_batch = batch.unique()

        # Node transformation
        data = self.NN_node(raw_x)

        # Compute edge embeddings using original node features
        hidden_edge_input = torch.cat((
            raw_x[edge_index[0]] * raw_x[edge_index[1]],  # Element-wise product
            raw_x[edge_index[0]],                         # Source node features
            raw_x[edge_index[1]]                          # Target node features
        ), dim=1)
        hidden_edge = self.NN_edge(hidden_edge_input).tanh()

        # Save intermediate data for debugging/visualization
        self.data = data
        self.model_edge = hidden_edge
        self.edge = torch.median(hidden_edge, dim=1)[0]

        # First GAT layer with edge attention
        data, atten_w1 = self.NN_conv1(data, edge_index, edge_attr=hidden_edge, return_attention_weights=True)
        data_1 = self.leaky(self.NN_flatten1(data))

        # Second GAT layer using a transformed edge index (edge_tf)
        data_2, atten_w2 = self.NN_conv2(data_1, edge_tf, return_attention_weights=True)
        data_2 = self.leaky(self.NN_flatten2(data_2))

        # Combine first and second GAT outputs
        data = data_1 + data_2

        self.atten_data = data

        # Apply residual transformation
        data = self.NN_res(data)
        self.res_data = data

        # Compute output only for nodes where gene_id_vec == 1
        gene_out = -F.log_softmax(data[gene_id_vec == 1], dim=1)[:, 0]
        return gene_out, atten_w1



# Set random seed for reproducibility
def set_seed(seed=42):
    # Set Python built-in random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed for CPU
    torch.manual_seed(seed)
    
    # Set PyTorch random seed for current GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs

    # Ensure deterministic behavior in cuDNN for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable benchmark to ensure deterministic results


def train_model(model, train_graph, num_epoch=5, batch_size=12, lr=0.0001, max_grad_norm=1.0, sparse_loss_weight=0.1, if_zero=False, use_device='cuda:0'):
    device = use_device if torch.cuda.is_available() else 'cpu'
    model.to(device)
    loss_exp = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_sparse1 = EdgeDiversityLoss1(diversity_weight=1)
    criterion_sparse2 = EdgeDiversityLoss1(diversity_weight=1)
    for epoch in range(num_epoch):
        train_loader = DataLoader(train_graph, batch_size=batch_size, shuffle=True)
        model.train()
        running_loss = 0.0
        running_loss1 = 0.0
        running_sparse_loss = 0.0 
        running_atten_loss = 0.0
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
            if if_zero == False:
                index = torch.where(sample.x[sample.id_vec == 1] > 0)[0]
                loss1 = loss_exp(gene_pre.flatten()[index], sample.y_exp.to(device)[index]) 
            else:
                index = torch.where(sample.x[sample.id_vec == 1])[0]
                loss1 = loss_exp(gene_pre.flatten(), sample.y_exp.to(device)) 
            temp_var_edge = torch.stack(torch.split(edge_temp, sample.edge_index.shape[1] // len(sample.y)))
            temp_var_atten = torch.stack(torch.split(atten[1], sample.edge_index.shape[1] // len(sample.y)))
            loss2 = sparse_loss_weight * criterion_sparse1(temp_var_edge) 
            loss3 = torch.tensor(0) 
            loss = loss1 + loss2 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            running_loss += loss.item()
            running_loss1 += loss1.item()
            running_sparse_loss += loss2.item()  
            running_atten_loss += loss3.item()  

            progress_bar.set_postfix(
                loss=running_loss / (progress_bar.n + 1),
                loss1=running_loss1 / (progress_bar.n + 1),
                sparse_loss=running_sparse_loss / (progress_bar.n + 1)
            )
            torch.cuda.empty_cache()
    
        print(f"Epoch [{epoch+1}/{num_epoch}], Average Loss: {running_loss / len(train_loader):.4f}, " 
              f"Loss1: {running_loss1 / len(train_loader):.4f}, Sparse Loss: {running_sparse_loss / len(train_loader):.4f}")

    return model

def train_model_sample(model, train_graph, sample_size=500, num_epoch=5, batch_size=12, lr=0.0001, max_grad_norm=1.0, sparse_loss_weight=0.1, use_device='cuda:0'):
    
    device = use_device if torch.cuda.is_available() else 'cpu'
    model.to(device)
    loss_exp = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_sparse1 = EdgeDiversityLoss1(diversity_weight=1)
    criterion_sparse2 = EdgeDiversityLoss1(diversity_weight=1)
    for epoch in range(num_epoch):
        train_loader = DataLoader(train_graph, batch_size=batch_size, shuffle=True)
        model.train()
        running_loss = 0.0
        running_loss1 = 0.0
        running_sparse_loss = 0.0 
        running_atten_loss = 0.0
        progress_bar = tqdm(train_loader, total = sample_size // batch_size, desc=f"Epoch {epoch+1}/{num_epoch}", unit="batch")
        for idx, sample in enumerate(progress_bar):
            if (idx - 1) * batch_size > sample_size:
                break
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
            temp_var_atten = torch.stack(torch.split(atten[1], sample.edge_index.shape[1] // len(sample.y)))
            loss2 = sparse_loss_weight * criterion_sparse1(temp_var_edge) 
            loss3 = torch.tensor(0) 
            loss = loss1 + loss2 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            running_loss += loss.item()
            running_loss1 += loss1.item()
            running_sparse_loss += loss2.item()  
            running_atten_loss += loss3.item()  

            progress_bar.set_postfix(
                loss=running_loss / (progress_bar.n + 1),
                loss1=running_loss1 / (progress_bar.n + 1),
                sparse_loss=running_sparse_loss / (progress_bar.n + 1)
            )
            torch.cuda.empty_cache()
    
        print(f"Epoch [{epoch+1}/{num_epoch}], Average Loss: {running_loss / len(train_loader):.4f}, " 
              f"Loss1: {running_loss1 / len(train_loader):.4f}, Sparse Loss: {running_sparse_loss / len(train_loader):.4f}")

    return model


def test_model(dataset_atac, model, test_graph, batch_size=1, device='cuda:2', if_test=False):
    """
    Run the model in test mode and collect output data.

    Parameters:
    - dataset_atac: the input dataset object
    - model: the trained model to evaluate
    - test_graph: test dataset (PyTorch Geometric format)
    - batch_size: number of samples per batch
    - device: computation device (e.g., 'cuda:2' or 'cpu')
    - if_test: flag to control the return format (used for testing/debugging)

    Returns:
    - result: aggregated model results if if_test is False
    - test_barcodes: list of cell barcodes corresponding to results
    OR
    - cell_link_atten: raw attention outputs per sample (if if_test=True)
    - cell_link_edge: edge values per sample
    - test_barcodes: barcodes corresponding to samples
    """

    # Create DataLoader for the test dataset
    test_loader = DataLoader(test_graph, batch_size=batch_size, shuffle=False, pin_memory=True)

    model.eval()        # Set model to evaluation mode
    model.to(device)    # Move model to specified device

    # Initialize result containers
    cell_type = []
    test_barcodes = []
    cell_link_atten = []
    cell_link_edge = []

    # Disable gradient computation for inference
    with torch.no_grad():
        for idx, sample in tqdm(enumerate(test_loader), total=len(test_loader), desc="Processing Batches"):
            gene_num = sample.y_exp.shape[0]

            # Forward pass through the model
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

            # Store barcodes and labels (on CPU)
            test_barcodes.extend(sample.cell)
            cell_type.extend(sample.y.cpu().numpy())

            # Process edge outputs
            edge_temp = model.edge.flatten().to('cpu')
            edge_lists = edge_temp.split(edge_temp.size(0) // sample_size)
            for edge in edge_lists:
                cell_link_edge.append(edge)

            # Process attention outputs
            atten_x_indices = atten[0][1].cpu()
            atten_enhancer_indices = atten[0][0].cpu()
            flattened_sample_x = sample.x[atten_x_indices].flatten().to('cpu')
            flattened_sample_enhancer = sample.x[atten_enhancer_indices].flatten().to('cpu')

            # Extract attention weights (mean or specific head)
            atten1_max = atten[1][:, 0].to('cpu')

            # Split attention data per sample and store
            for x, enhancer, atten_ in zip(
                flattened_sample_x.split(flattened_sample_x.size(0) // sample_size),
                flattened_sample_enhancer.split(flattened_sample_enhancer.size(0) // sample_size),
                atten1_max.split(atten1_max.size(0) // sample_size)
            ):
                cell_link_atten.append(atten_)

            # Release unused GPU memory
            torch.cuda.empty_cache()

    # Stack attention and edge tensors
    cell_link_atten = torch.stack(cell_link_atten)
    cell_link_edge = torch.stack(cell_link_edge)

    # Calculate edge count scaling factors
    edge_df = get_edge_info(dataset_atac)
    gene_counts = edge_df['gene'].value_counts()
    gene_edge_counts_dict = gene_counts.to_dict()
    gene_list = [dataset_atac.array_peak[t] for t in test_graph[0].edge_index[1]]
    edge_counts = [gene_edge_counts_dict[t] for t in gene_list]
    edge_counts = np.array(edge_counts)

    # Combine attention, edge, and gene interaction counts
    result = cell_link_edge.cpu().numpy() * cell_link_atten.cpu().numpy() * edge_counts

    # Return format based on mode
    if if_test:
        return cell_link_atten, cell_link_edge, test_barcodes
    else:
        return result, test_barcodes


def process_samples(dataset_atac, test_cell):
    """
    Process sample data and split into training and testing sets.

    Parameters:
    - dataset_atac: ATAC-seq dataset object containing peaks and graphs
    - test_cell: list of cell identifiers to be used as test samples

    Returns:
    - train_graph: list of graph samples for training
    - test_graph: list of graph samples for testing
    - train_cell_type: list of training cell labels
    - test_cell_type: list of testing cell labels
    """
    data = dataset_atac.array_peak
    
    # Create a torch vector to mark peaks starting with 'chr' as 0, others as 1
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
    dataset_graph = dataset_atac.list_graph
    
    # Iterate over all graph samples in the dataset
    for i, sample in tqdm(enumerate(dataset_graph), total=len(dataset_graph), desc='Processing samples'):
        sample.id_vec = torch_vector   # Assign the id vector to each sample
        sample.seq_data = 1            # Mark seq_data flag (could represent data type or state)
        
        # Split samples into test or train based on cell identifier membership
        if sample.cell in test_cell:
            test_graph.append(sample)
            test_cell_type.append(sample.y.item())
        else:
            train_graph.append(sample)
            train_cell_type.append(sample.y.item())

    return train_graph, test_graph, train_cell_type, test_cell_type


def process_samples_add_random(train_graph_1, torch_vector, max_samples=1000):
    """
    Further process training graph data by shuffling and randomizing features.

    Parameters:
    - train_graph_1: initial list of training graph samples
    - torch_vector: vector used for identification (same as in process_samples)
    - max_samples: maximum number of samples to process

    Returns:
    - train_graph: processed training graph samples with randomized features
    - train_cell_type: corresponding labels for training samples
    """
    dataset_graph = train_graph_1
    random.shuffle(dataset_graph)  # Shuffle samples randomly
    
    train_graph = []
    train_cell_type = []
    
    # Find max value of features in the training dataset for scaling randomization
    max_values = 0
    for t in dataset_graph:
        temp = torch.max(t.x)
        if temp > max_values:
            max_values = temp
    max_values = max_values.item()
    print('random max_values:', max_values)
    
    # Process samples up to max_samples
    for i, sample in tqdm(enumerate(dataset_graph), total=min(len(dataset_graph), max_samples), desc='Processing samples'):
        if i == max_samples:
            break
        
        sample.id_vec = torch_vector    # Assign id vector
        sample.seq_data = 2             # Mark seq_data flag differently (maybe for augmentation)
        
        # Randomize features in sample.x scaled by max_values
        sample.x = torch.rand_like(sample.x) * max_values
        
        # Shuffle expression labels in y_exp
        sample.y_exp = sample.y_exp[torch.randperm(sample.y_exp.size(0))]
        
        sample.y = torch.tensor([-1])   # Reset label to -1 (unlabeled or unknown)
        
        train_graph.append(sample)
        train_cell_type.append(sample.y.item())

    return train_graph, train_cell_type


import torch
import random
from collections import Counter
from tqdm import tqdm

def balance_classes(train_graph, train_cell_type, seed=42):
    """
    Balance the number of samples across different classes in the training set.

    Parameters:
    - train_graph: list of training graph samples
    - train_cell_type: list of corresponding cell type labels for training samples
    - seed: random seed for reproducibility (default is 42)

    Returns:
    - balanced_train_graph: training graph samples after balancing classes
    - balanced_train_cell_type: updated list of cell type labels after balancing
    """

    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # Count the number of samples for each class
    class_counts = Counter(train_cell_type)

    # Determine the maximum sample count among all classes
    max_count = max(class_counts.values())

    balanced_train_graph = []

    # For each class, duplicate samples to reach max_count
    for cell_type, count in class_counts.items():
        # Collect samples belonging to the current class
        samples = [sample for sample in train_graph if sample.y.item() == cell_type]

        # Calculate how many samples need to be added to balance
        samples_to_add = max_count - count

        # Add original samples plus randomly chosen duplicates to balanced dataset
        balanced_train_graph.extend(samples)
        balanced_train_graph.extend(random.choices(samples, k=samples_to_add))

    # Update the cell type list based on the balanced training graph
    balanced_train_cell_type = [sample.y.item() for sample in balanced_train_graph]

    return balanced_train_graph, balanced_train_cell_type


def preprocess_graph(dataset_atac, test_cell, max_random_sample=500, seed=42, if_test=False, is_balances=True):
    if is_balances:
        print('Preprocess Start!')
        train_graph_1, test_graph, train_cell_type_1, test_cell_type = process_samples(dataset_atac, test_cell)
        print('train_graph:', len(train_graph_1))
        print('test_graph:', len(test_graph))

        torch_vector = train_graph_1[0].id_vec

        balanced_train_graph, balanced_train_cell_type = balance_classes(train_graph_1, train_cell_type_1, seed=seed)
        print('balanced_train_graph:', len(balanced_train_graph))

        train_graph_2, train_cell_type_2 = process_samples_add_random(train_graph_1, torch_vector, max_samples=max_random_sample)
        print('adding random nodes...:', len(train_graph_2))

        balanced_train_graph = balanced_train_graph + train_graph_2
        balanced_train_cell_type = balanced_train_cell_type + train_cell_type_2
        if if_test==False:
            return balanced_train_graph, test_graph
        else:
            return train_graph_1, test_graph
    else:
        print('Preprocess Start!')
        train_graph_1, test_graph, train_cell_type_1, test_cell_type = process_samples(dataset_atac, test_cell)
        print('train_graph:', len(train_graph_1))
        print('test_graph:', len(test_graph))
        torch_vector = train_graph_1[0].id_vec
        train_graph_2, train_cell_type_2 = process_samples_add_random(train_graph_1, torch_vector, max_samples=max_random_sample)
        print('adding random nodes...:', len(train_graph_2))
        if if_test==False:
            temp_train_graph = train_graph_1 + train_graph_2
            return temp_train_graph, test_graph
        else:
            return train_graph_1, test_graph


import pandas as pd
import numpy as np
from collections import defaultdict

def process_data(dataset_atac, file_tss, file_peaks_promoter, flank_proximal=2000):
    """
    Process data to find the nearest peak for each gene.

    Parameters:
    - dataset_atac: ATAC-seq dataset object containing peak information
    - file_tss: path to the gene TSS (Transcription Start Site) file
    - file_peaks_promoter: path to the peaks_promoter file containing peak-gene associations
    - flank_proximal: upstream and downstream range (in base pairs) around TSS for proximal region (default is 2000)

    Returns:
    - anchor_peak_dict: dictionary mapping each gene to its nearest peak
    """

    # Extract gene list from dataset, filtering out entries starting with 'chr'
    gene_list = [t for t in dataset_atac.array_peak if not t.startswith('chr')]

    # Load TSS data into a DataFrame
    df_tss = pd.read_csv(file_tss, sep='\t', header=None)
    df_tss.columns = ['chrom', 'tss', 'symbol', 'ensg_id', 'strand']

    # Remove duplicate genes, keeping only unique symbols
    df_tss = df_tss.drop_duplicates(subset='symbol')

    # Set gene symbol as DataFrame index for quick access
    df_tss.index = df_tss['symbol']

    # Calculate TSS window ±2000 bp for promoter regions
    df_tss['tss_start'] = df_tss['tss'] - 2000
    df_tss['tss_end'] = df_tss['tss'] + 2000

    # Calculate proximal region window using flank_proximal parameter
    df_tss['proximal_start'] = df_tss['tss'] - flank_proximal
    df_tss['proximal_end'] = df_tss['tss'] + flank_proximal

    # Extract promoter and proximal region DataFrames
    df_promoter = df_tss.loc[:, ['chrom', 'tss_start', 'tss_end', 'symbol', 'ensg_id', 'strand']]
    df_proximal = df_tss.loc[:, ['chrom', 'proximal_start', 'proximal_end', 'symbol', 'ensg_id', 'strand']]

    # Dictionary to store peaks associated with each gene and their distances to TSS
    dict_promoter = defaultdict(list)

    # Parse the peaks_promoter file line by line
    with open(file_peaks_promoter, 'r') as w_pro:
        for line in w_pro:
            list_line = line.strip().split('\t')
            # Skip entries with '.' in the 5th column (index 4)
            if list_line[4] == '.':
                continue
            gene_symbol = list_line[7]
            peak = list_line[3]

            # Get TSS coordinate for the gene
            gene_tss = df_tss.loc[gene_symbol, 'tss']

            # Calculate the midpoint coordinate of the peak (CRE)
            coor_cre = (int(list_line[2]) + int(list_line[1])) / 2

            # Calculate distance between gene TSS and peak midpoint
            dist_gene_cre = abs(gene_tss - coor_cre)

            # Append peak and its distance to the gene's list
            dict_promoter[gene_symbol].append((peak, dist_gene_cre))

    # For each gene, find the nearest peak by minimal distance
    anchor_peak_dict = {}
    for gene in gene_list:
        promoter_res = dict_promoter[gene]
        if len(promoter_res) == 0:
            # If no peaks found, assign None
            anchor_peak_dict[gene] = None
        else:
            # Find the peak with minimum distance to TSS
            distances = [t[1] for t in promoter_res]
            min_index = np.argmin(distances)
            anchor_peak_dict[gene] = promoter_res[min_index][0]

    return anchor_peak_dict



def get_peaks_df(dataset_atac):
    peaks_list = [t for t in dataset_atac.array_peak if t.startswith('chr')]
    source_df = pd.DataFrame( [peak.split('-') for peak in peaks_list], columns=['Chromosome', 'Start', 'End'] )
    return source_df


def compute_summed_peaks(selected_tensor, cell_types, num=10):
    """
    Compute the summed peaks for each cell type.

    Parameters:
    selected_tensor (torch.Tensor): A tensor of shape (num_cells, num_features).
    cell_types (pandas.Series): Cell type labels corresponding to the rows of selected_tensor.

    Returns:
    torch.Tensor: A tensor containing the summed peaks for each cell.
    """
    # Convert cell types to a numpy array for easier indexing
    cell_types = cell_types.to_numpy()
    
    # Initialize an empty list to store the summed peaks results
    summed_peaks = []
    
    # Iterate over each unique cell type
    for cell_type in np.unique(cell_types):
        # Find indices of cells belonging to the current cell type
        type_indices = np.where(cell_types == cell_type)[0]
        
        # Iterate over each cell index within the current cell type
        for idx in type_indices:
            # Extract the tensor row corresponding to the current cell
            current_cell = selected_tensor[idx]
            
            # Randomly sample 'num' other cells of the same type (excluding current cell)
            other_cells_indices = np.random.choice(type_indices[type_indices != idx], size=num, replace=False)
            
            # Extract tensor rows for the sampled other cells
            other_cells = selected_tensor[other_cells_indices]
            
            # Compute the sum of the current cell and the sampled other cells along the feature dimension
            summed_peak = torch.sum(torch.cat([current_cell.unsqueeze(0), other_cells], dim=0), dim=0)
            
            # Append the summed peak tensor to the results list
            summed_peaks.append(summed_peak)
    
    # Stack the list of summed peaks into a single tensor
    summed_peaks_tensor = torch.stack(summed_peaks)
    
    return summed_peaks_tensor


def generate_peaks_list(dataset_atac, anchor_peak_dict):
    """
    Generate a list of peaks based on dataset_atac's array_peak,
    replacing gene names with their corresponding anchor peak from the dictionary.

    Parameters:
    dataset_atac: An object containing the attribute 'array_peak', a list of peaks.
    anchor_peak_dict: A dictionary mapping gene names to their nearest peak (anchor peak).

    Returns:
    list: A list of peaks where gene names are replaced by their anchor peaks.
    """
    peaks_list = []
    for t in dataset_atac.array_peak:
        if not t.startswith('chr'):
            # Replace gene name with its anchor peak from the dictionary
            peaks_list.append(anchor_peak_dict[t])
        else:
            # Keep the original peak if it starts with 'chr'
            peaks_list.append(t)
    return peaks_list


import torch
from tqdm import tqdm

def expand_features(summed_mt, overlap_peaks, peak_list):
    """
    Expand the summed_mt matrix to include all features in peak_list.
    For features not in overlap_peaks, add zero-filled columns.

    Parameters:
    summed_mt (torch.Tensor): Input matrix of shape (n_cells, n_features1).
    overlap_peaks (list): List of feature names corresponding to summed_mt columns.
    peak_list (list): Target list of all desired feature names.

    Returns:
    torch.Tensor: Expanded matrix of shape (n_cells, len(peak_list)).
    """
    # Create a dictionary mapping each feature in overlap_peaks to its column index
    overlap_peaks_idx = {peak: idx for idx, peak in enumerate(overlap_peaks)}
    
    # Initialize an output matrix filled with zeros
    n_cell = summed_mt.shape[0]
    expanded_mt = torch.zeros((n_cell, len(peak_list)), dtype=summed_mt.dtype)
    
    # Fill the expanded matrix with values from summed_mt for overlapping features
    for col_idx, peak in tqdm(enumerate(peak_list), desc="Expanding features", total=len(peak_list)):
        if peak in overlap_peaks_idx:
            expanded_mt[:, col_idx] = summed_mt[:, overlap_peaks_idx[peak]]
    
    return expanded_mt, peak_list


def process_genes_and_select_rows(rna_ref, ref_adata, gene_list):
    """
    Process gene list and select rows from rna_ref according to cell types in ref_adata.

    Parameters:
    rna_ref (pd.DataFrame): RNA reference dataframe.
    ref_adata (anndata.AnnData): Reference AnnData object containing 'celltype' info in .obs.
    gene_list (list): List of genes to retain.

    Returns:
    torch.Tensor: Processed feature matrix corresponding to ref_adata cell types and gene_list.
    """
    # Create an empty DataFrame with columns strictly ordered by gene_list
    filtered_rna_ref = pd.DataFrame(columns=gene_list)
    
    # For each gene in gene_list, copy column if exists or fill with zeros
    for gene in tqdm(gene_list, desc="Processing genes"):
        if gene in rna_ref.columns:
            filtered_rna_ref[gene] = rna_ref[gene]
        else:
            filtered_rna_ref[gene] = 0
    
    # Update rna_ref to the filtered DataFrame
    rna_ref = filtered_rna_ref
    
    # Get cell type labels from ref_adata.obs
    temp_celltypes = ref_adata.obs['celltype']
    
    # Select rows from filtered_rna_ref that correspond to the cell types in ref_adata
    selected_rows = filtered_rna_ref.loc[temp_celltypes]
    
    # Convert selected rows to a torch tensor and return
    ref_nodes_exp = torch.tensor(selected_rows.values)
    
    return ref_nodes_exp




def map_celltypes_to_unique_numbers(celltypes, existing_labels):
    """
    Map celltypes to unique numeric labels that do not overlap with existing labels.

    Parameters:
    celltypes (np.ndarray): Array of cell types to be mapped.
    existing_labels (list): List of existing numeric labels.

    Returns:
    np.ndarray: Array of mapped numeric labels corresponding to celltypes.
    """
    # Find the maximum label in existing_labels
    max_label = max(existing_labels)

    # Generate new unique labels starting from max_label + 1
    new_labels = np.arange(max_label + 1, max_label + 1 + len(set(celltypes)))

    # Create a mapping from each unique celltype to a new unique label
    unique_celltypes = set(celltypes)
    celltype_to_number = {celltype: new_label for celltype, new_label in zip(unique_celltypes, new_labels)}

    # Map each celltype in the input array to its new label
    mapped_labels = np.array([celltype_to_number[celltype] for celltype in celltypes])

    return mapped_labels



def create_ref_graph(ref_adata, ref_nodes_mt, ref_nodes_edge_index, mapped_labels, ref_nodes_edge_tf, ref_nodes_exp, ref_nodes_barcodes, ref_nodes_id_vec):
    ref_graph = []
    for i in range(ref_adata.shape[0]):
        data = Data(
            x=ref_nodes_mt[i],
            edge_index=ref_nodes_edge_index,
            y=torch.tensor([mapped_labels[i]]),
            edge_tf=ref_nodes_edge_tf,
            y_exp=ref_nodes_exp[i],
            cell=ref_nodes_barcodes[i],
            id_vec=ref_nodes_id_vec,
            seq_data=1
        )
        ref_graph.append(data)
    return ref_graph


def Create_ref_graph(graph, dataset_atac, expanded_mt, ref_adata, rna_ref, gene_list):
    data = dataset_atac.array_peak
    graph_temp = graph[0]
    torch_vector = torch.zeros(len(data))
    for idx, item in enumerate(data):
        if item.startswith('chr'):
            torch_vector[idx] = 0 
        else:
            torch_vector[idx] = 1 
    existing_labels = list(set([t.y.item() for t in graph]))
    mapped_labels = map_celltypes_to_unique_numbers(ref_adata.obs['celltype'].values, existing_labels)
    ref_nodes_edge_index = graph_temp.edge_index.to(torch.int64)
    ref_nodes_edge_tf = graph_temp.edge_tf.to(torch.int64)
    ref_nodes_id_vec = torch_vector
    ref_seq_data = 1
    
    log2p_expanded_mt = torch.log2(expanded_mt + 1)
    ref_nodes_mt = torch.unsqueeze(log2p_expanded_mt, dim=2).to(torch.float32)
    ref_nodes_barcodes = list(ref_adata.obs.index)
    ref_nodes_exp = process_genes_and_select_rows(rna_ref, ref_adata, gene_list).to(torch.float32)
    row_sums = ref_nodes_exp.sum(dim=1, keepdim=True)
    ref_nodes_exp = ref_nodes_exp / row_sums
     
    ref_graph = []
    for i in range(ref_adata.shape[0]):
        data = Data(x=ref_nodes_mt[i], edge_index=ref_nodes_edge_index, y=torch.tensor([mapped_labels[i]]), edge_tf=ref_nodes_edge_tf, y_exp=ref_nodes_exp[i], 
                    cell=ref_nodes_barcodes[i], id_vec=ref_nodes_id_vec, seq_data=1)
        ref_graph.append(data)
    return ref_graph


def get_overlap_peaks(ref_peaks_df, source_peaks_df):
    """
    Get overlapping peaks from ref_peaks_df and source_peaks_df and generate Combined columns.

    Parameters:
    ref_peaks_df (pd.DataFrame): Reference peaks DataFrame.
    source_peaks_df (pd.DataFrame): Source peaks DataFrame.

    Returns:
    np.ndarray: Values of the Combined column containing overlapping peaks.
    """
    # Convert DataFrames to BedTool objects
    ref_bed = pybedtools.BedTool.from_dataframe(ref_peaks_df)
    source_bed = pybedtools.BedTool.from_dataframe(source_peaks_df)
    
    # Get intersection of the two BedTool objects
    intersection = ref_bed.intersect(source_bed, wa=True, wb=True)
    
    # Convert intersection result to DataFrame
    result_df = intersection.to_dataframe(names=[
        'Chromosome', 'Start', 'End', 
        'source_Chromosome', 'source_Start', 'source_End'
    ])
    
    # Create Combined columns for source and reference peaks
    result_df['Combined'] = result_df.apply(
        lambda row: f"{row['source_Chromosome']}-{row['source_Start']}-{row['source_End']}", axis=1
    )
    result_df['Combined_ref'] = result_df.apply(
        lambda row: f"{row['Chromosome']}-{row['Start']}-{row['End']}", axis=1
    )
    
    # Extract Combined column values
    overlap_peaks = result_df['Combined'].values
    overlap_ref_peaks = result_df['Combined_ref'].values
    
    return overlap_peaks, overlap_ref_peaks


def save_or_load_weights(model, file_path):
    """
    Load model weights if the file exists; otherwise, save the model weights.

    Parameters:
    model (torch.nn.Module): The model instance.
    file_path (str): Path to the weight file.
    """
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path))
        print(f"Model weights loaded from {file_path}")
    else:
        torch.save(model.state_dict(), file_path)
        print(f"Model weights saved to {file_path}")


def process_edge_names(edge_names_path, dataset_atac):
    """
    Process edge names file and return a list of edge indices.

    Parameters:
    edge_names_path (str): Path to the edge names file.
    dataset_atac (object): Dataset object containing 'array_peak' and 'list_graph'.

    Returns:
    list: List of edge indices corresponding to the processed edge names.
    """
    # Read edge names from file
    edge_names = pd.read_csv(edge_names_path, sep='\t', index_col=0).index
    edge_list = []

    # Parse and process edge names into gene-region format
    for t in edge_names:
        split_res = t.split("'")
        gene_tag = split_res[1]
        region = split_res[3]
        edge_list.append(gene_tag + '-' + region)

    # Extract edges from dataset_atac graph
    cols1 = dataset_atac.list_graph[0].edge_index[0, :]
    cols2 = dataset_atac.list_graph[0].edge_index[1, :]
    tag1 = [dataset_atac.array_peak[t] for t in cols2]
    tag2 = [dataset_atac.array_peak[t] for t in cols1]
    edge_names_anchors = [t1 + '-' + t2 for t1, t2 in zip(tag1, tag2)]

    # Map processed edge names to their indices in the graph
    edge_index = []
    for t in edge_list:
        try:
            edge_index.append(edge_names_anchors.index(t))
        except ValueError:
            # Skip if edge name not found
            continue
    
    return edge_index




def test_model_for_gene_loss(model, test_graph, batch_size=1, device='cuda:2', if_test=False):
    """
    Run the model testing process and collect result data.

    Parameters:
    - model: The model to be tested.
    - test_graph: The test dataset.
    - batch_size: Batch size for testing.
    - device: Device to run the model on ('cuda:2' or 'cpu').
    - if_test: Boolean flag to determine the return type.

    Returns:
    - If if_test is True:
        Returns cell_link_atten, cell_link_edge, test_barcodes, and loss1_list.
      Otherwise:
        Returns the combined result matrix, test_barcodes, and loss1_list.
    """
    loss_exp = torch.nn.MSELoss()
    test_loader = DataLoader(test_graph, batch_size=batch_size, shuffle=False, pin_memory=True)
    model.eval()
    model.to(device)

    cell_type = []
    test_barcodes = []
    cell_link_atten = []
    cell_link_edge = []
    loss1_list = []

    with torch.no_grad():
        for idx, sample in tqdm(enumerate(test_loader), total=len(test_loader), desc="Processing Batches"):
            gene_num = sample.y_exp.shape[0]

            # Forward pass through the model
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
            
            # Calculate loss
            loss1 = loss_exp(gene_pre.flatten(), sample.y_exp.to(device)) 
            loss1_list.append(loss1.item())

            sample_size = len(sample.y)
            test_barcodes.extend(sample.cell)  # Keep on CPU
            cell_type.extend(sample.y.cpu().numpy())

            # Process edge data
            edge_temp = model.edge.flatten().to('cpu')
            edge_lists = edge_temp.split(edge_temp.size(0) // sample_size)
            for edge in edge_lists:
                cell_link_edge.append(edge)

            # Process attention data
            atten_x_indices = atten[0][1].cpu()
            atten_enhancer_indices = atten[0][0].cpu()
            flattened_sample_x = sample.x[atten_x_indices].flatten().to('cpu')
            flattened_sample_enhancer = sample.x[atten_enhancer_indices].flatten().to('cpu')
            atten1_max = atten[1].mean(dim=1).to('cpu')

            # Split and collect attention values per cell
            for x, enhancer, atten_ in zip(
                flattened_sample_x.split(flattened_sample_x.size(0) // sample_size),
                flattened_sample_enhancer.split(flattened_sample_enhancer.size(0) // sample_size),
                atten1_max.split(atten1_max.size(0) // sample_size)
            ):
                cell_link_atten.append(atten_)

            torch.cuda.empty_cache()

    # Stack collected tensors
    cell_link_atten = torch.stack(cell_link_atten)
    cell_link_edge = torch.stack(cell_link_edge)

    # Get edge information and counts
    edge_df = get_edge_info(dataset_atac)
    gene_counts = edge_df['gene'].value_counts()
    gene_edge_counts_dict = gene_counts.to_dict()
    gene_list = [dataset_atac.array_peak[t] for t in test_graph[0].edge_index[1]]
    edge_counts = [gene_edge_counts_dict[t] for t in gene_list]
    edge_counts = np.array(edge_counts)

    # Calculate final result matrix
    result = cell_link_edge.cpu().numpy() * cell_link_atten.cpu().numpy() * edge_counts

    if if_test:
        return cell_link_atten, cell_link_edge, test_barcodes, loss1_list
    else:
        return result, test_barcodes, loss1_list



def get_gene_expression(dataset_atac, model, test_graph, batch_size=1, device='cuda:2'):
    """
    Run model testing and return gene expression predictions.

    Returns:
    - gene_expression: [num_cells, num_genes]
    - test_barcodes: list of barcodes
    - cell_type: list of cell types
    """
    test_loader = DataLoader(test_graph, batch_size=batch_size, shuffle=False, pin_memory=True)
    model.eval()
    model.to(device)

    cell_type = []
    test_barcodes = []
    gene_expression = []

    with torch.no_grad():
        for idx, sample in tqdm(enumerate(test_loader), total=len(test_loader), desc="Processing Batches"):
            gene_num = sample.y_exp.shape[0]
            # Model prediction
            pred_expression, _ = model(
                sample.seq_data.to(device),
                sample.x.to(device),
                sample.edge_index.to(device),
                sample.edge_tf.T.to(device),
                sample.batch.to(device),
                gene_num,
                sample.id_vec.to(device),
                is_test=True
            )
            # Determine shape from prediction
            cell_num = sample.y.shape[0]  # Number of cells in this batch
            gene_num = pred_expression.shape[0] // cell_num
            
            # Reshape to (cell_num, gene_num)
            pred_expression = pred_expression.view(cell_num, gene_num)

            # Store
            test_barcodes.extend(sample.cell)
            cell_type.extend(sample.y.cpu().numpy())
            gene_expression.append(pred_expression.cpu())

            torch.cuda.empty_cache()

    # Concatenate all cells (dim=0 means stacking rows)
    gene_expression = torch.cat(gene_expression, dim=0)

    return gene_expression.numpy(), test_barcodes, cell_type


    
def get_edge_info(dataset_atac):
    peak_index = dataset_atac.list_graph[0].edge_index[0]
    gene_index = dataset_atac.list_graph[0].edge_index[1]
    peak_list = [dataset_atac.array_peak[t] for t in peak_index]
    gene_list = [dataset_atac.array_peak[t] for t in gene_index]
    return pd.DataFrame({'peak': peak_list, 'gene': gene_list})

def get_edge_info_all(df_edges):
    test_edges = dataset_atac.list_graph[0].edge_index[:, df_edges['names'].values.astype(int)]
    peak_list = [dataset_atac.array_peak[t] for t in test_edges[0]]
    gene_list = [dataset_atac.array_peak[t] for t in test_edges[1]]
    edge_info = pd.DataFrame({'peak': peak_list, 'gene': gene_list})
    edge_info['peak_gene'] = edge_info['peak'] + ':' + edge_info['gene']
    df_edges['pairs'] = edge_info['peak_gene'].values
    df_edges['gene'] = edge_info['gene'].values
    df_edges['peak'] = edge_info['peak'].values
    print(df_edges.shape)
    return df_edges

def get_edge_info_with_anchor(dataset_atac, anchor_peaks):
    peak_index = dataset_atac.list_graph[0].edge_index[0]
    gene_index = dataset_atac.list_graph[0].edge_index[1]
    peak_list = [dataset_atac.array_peak[t] for t in peak_index]
    gene_list = [dataset_atac.array_peak[t] for t in gene_index]
    promoter_list = [anchor_peak[t] for t in gene_list]
    return pd.DataFrame({'peak': peak_list, 'gene': gene_list, 'promoter': promoter_list})