# scReGAT

This repository contains the code for scReGAT, a deep learning framework for predicting long-range gene regulation at single-cell resolution, as described in our paper:

**Leveraging knowledge of regulatory interactions to predicting long-range gene regulation at single cell resolution with scReGAT** .

## Introduction

Understanding gene regulation at single-cell resolution is crucial for unraveling the complexities of development, disease, and cellular identity. scReGAT (single-cell Regulatory Graph Attention Network) is a deep learning framework that integrates prior knowledge of cis-regulatory element (cRE)-gene and transcription factor (TF)-gene interactions to reconstruct cell-specific regulatory networks.

At the core of scReGAT is a knowledge-guided regulatory graph (kRG), which combines experimentally validated regulatory interactions with cell-resolved chromatin accessibility profiles. This graph serves as the foundation for training a Graph Attention Network (GAT) that predicts gene expression and quantifies the contribution of specific regulatory interactions using an interpretable regulatory probability (RP) for each edge.

scReGAT has been benchmarked across five single-cell multi-omics datasets and has been shown to:

- Successfully recapitulate known cell-type-specific cRE-gene interactions.
- Uncover dynamic regulatory rewiring that predicts transcriptional transitions in neuroblastoma and osteogenic differentiation systems.
- Identify disease-associated cell types and uncover candidate regulatory mechanisms underlying complex trait associations by integrating GWAS loci.

## Architecture

The scReGAT framework is composed of two main stages: knowledge-guided regulatory graph (kRG) construction and a graph attention network (GAT) for gene expression prediction and regulatory interaction scoring.

![Architecture Image](docs/Fig1.png)

_Figure 1: An overview of the scReGAT framework, from data integration and graph construction to the GAT-based model for predicting gene expression and inferring regulatory probabilities._

## Installation

### Dependencies

First, create a conda environment with Python 3.10.

```bash
conda create -n scregat python=3.10 -y
conda activate scregat
```

Next, install the required deep learning libraries.
\*PyTorch:
This example is for CUDA 11.8. Please adjust the command according to your specific CUDA version. For more options, visit the [PyTorch website](https://pytorch.org/get-started/locally/).

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
```

PyG (PyTorch Geometric):
Follow the official instructions based on your PyTorch and CUDA versions: [PyG Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

```bash
pip install torch_geometric
```

### Install scReGAT

Clone the repository and install the package using pip.

```bash
git clone https://github.com/TianLab-Bioinfo/scReGAT.git
cd scReGAT
pip install .
```

## Qucik Start

Please see `notebook/train_model_MFG.ipynb` and `notebook/train_model_Pan.ipynb` for examples.

## Contact Us

If you encounter any issues during use or have any suggestions, feel free to contact us:

- Baole Wen: blwen24@m.fudan.edu.cn
- Yi Long : longy25@m.fudan.edu.cn
  You can also submit an issue on GitHub.

For more information about our research, please visit our lab website: [Tian Lab](https://tianlab-bioinfo.github.io/).

## Citation

If you use scReGAT in your research, please cite:

```

```
