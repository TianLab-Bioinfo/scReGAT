"""
SCReGAT Model Definition

Contains the SCReGAT neural network model and loss functions.
"""

import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


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


def set_seed(seed=42):
    """Set random seed for reproducibility."""
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

