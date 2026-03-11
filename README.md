# MiniTorch

A teaching-oriented deep learning systems project built around the MiniTorch framework. This repository contains my implementation and extensions across automatic differentiation, tensor operations, fast operators, convolutional neural networks, and end-to-end training for sentiment analysis and image classification.

## Overview

This project started from the MiniTorch student suite and was extended as a personal deep learning systems project. The core goal is to understand how a modern deep learning framework works under the hood, from scalar autodiff and computation graphs to batched tensor operators and convolution-based models.

The repository currently supports two usage styles:

- Coursework-style MiniTorch training scripts under `project/`, focused on understanding framework internals.
- Modern training scripts under `scripts/`, using PyTorch `DataLoader` workflows and Hugging Face `Trainer` for easier reproduction.

## Highlights

- Implemented and debugged core autodiff and tensor infrastructure, including backpropagation, broadcasting, parameter management, and tensor reshaping/permutation.
- Implemented neural network building blocks such as `Conv1D`, `Conv2D`, pooling, dropout, and `logsoftmax`.
- Completed end-to-end training pipelines for:
  - SST-2 sentiment classification
  - MNIST digit classification
- Added modern baseline training entrypoints to make the project easier to run with current PyTorch and Transformers tooling.
- Preserved the original MiniTorch educational workflow while adding more practical experiment scripts.

## Tech Stack

- Python 3.11
- MiniTorch
- NumPy
- Numba
- PyTorch
- Hugging Face Datasets / Transformers
- Streamlit / Plotly

## Repository Structure

```text
minitorch/
|- minitorch/                  # framework core: autodiff, tensor ops, fast ops, nn ops
|- project/                    # original coursework scripts and visualizations
|  |- run_sentiment.py         # MiniTorch-based SST-2 sentiment training
|  |- run_mnist_multiclass.py  # MiniTorch-based MNIST multiclass training
|  |- app.py                   # Streamlit visualization entrypoint
|  |- data/                    # local data directory (MNIST, optional caches)
|- scripts/                    # modern training entrypoints
|  |- train_sentiment_hf.py    # Hugging Face Trainer on SST-2
|  |- train_mnist_torch.py     # PyTorch DataLoader + AdamW on MNIST
|- tests/                      # unit and grad-check tests
|- requirements.txt            # base dependencies
|- requirements-modern.txt     # extra dependencies for modern training scripts
```

## Environment Setup

### Base environment

```bash
conda create -n myminitorch python=3.11
conda activate myminitorch

pip install -r requirements.txt
pip install -e .
```

### Optional modern training stack

```bash
pip install -r requirements-modern.txt
```

### Optional GPU PyTorch install

Install the proper PyTorch build for your CUDA version if you want GPU acceleration.

Example for CUDA 12.6:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## Data Preparation

### 1. MNIST for MiniTorch and PyTorch CNN scripts

The MNIST scripts read raw IDX files from `project/data/`.

Required files:

- `project/data/train-images-idx3-ubyte`
- `project/data/train-labels-idx1-ubyte`

On Linux:

```bash
mkdir -p project/data
cd project/data

wget -c https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
wget -c https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz

gunzip -kf train-images-idx3-ubyte.gz
gunzip -kf train-labels-idx1-ubyte.gz
cd ../..
```

### 2. SST-2 and GloVe for the original MiniTorch sentiment script

The coursework script `project/run_sentiment.py` uses:

- Hugging Face `datasets` cache for SST-2
- `embeddings` package cache for GloVe

Recommended environment variables on Linux:

```bash
export HF_HOME=/root/shared-nvme/minitorch/project/data/hf_cache
export EMBEDDINGS_ROOT=/root/shared-nvme/minitorch/project/data
```

With this setup:

- SST-2 cache will go under `project/data/hf_cache/`
- GloVe cache will go under `project/data/glove/`

Note: the `embeddings` package expects a zip file named `wikipedia_gigaword.zip` under `glove/`. If you already have extracted `glove.6B.*.txt` files, package the needed file back into a zip archive before running `project/run_sentiment.py`.

### 3. SST-2 for the modern Transformer script

The modern script `scripts/train_sentiment_hf.py` uses Hugging Face datasets and pretrained checkpoints directly. It does not require GloVe.

Recommended:

```bash
export HF_HOME=/root/shared-nvme/minitorch/project/data/hf_cache
```

## How to Run

### A. Original MiniTorch training scripts

Run from the repository root.

#### Sentiment classification

```bash
export HF_HOME=/root/shared-nvme/minitorch/project/data/hf_cache
export EMBEDDINGS_ROOT=/root/shared-nvme/minitorch/project/data

python project/run_sentiment.py | tee sentiment.txt
```

Expected log items:

- training loss
- training accuracy
- validation accuracy

#### MNIST multiclass classification

```bash
python project/run_mnist_multiclass.py | tee mnist.txt
```

Expected log items:

- training loss
- validation accuracy

### B. Modern training scripts

These scripts are easier to reproduce on current environments and are better suited for demonstration.

#### Hugging Face SST-2 training

```bash
export HF_HOME=/root/shared-nvme/minitorch/project/data/hf_cache

python scripts/train_sentiment_hf.py \
  --output-dir outputs/sst2-hf \
  --max-train-samples 2000 \
  --max-eval-samples 500
```

#### PyTorch MNIST training

```bash
python scripts/train_mnist_torch.py \
  --data-dir project/data \
  --output-dir outputs/mnist-torch \
  --epochs 5
```

### C. Streamlit visualization

```bash
streamlit run project/app.py
```

This interface can be used to inspect intermediate tensors, hidden states, and training behavior for selected modules.

## Recommended Result Artifacts

After training, it is useful to keep:

- `sentiment.txt`
- `mnist.txt`
- `outputs/sst2-hf/metrics.json`
- `outputs/mnist-torch/metrics.json`

These files make it easier to compare the educational MiniTorch pipeline with the modern PyTorch / Transformers pipeline.

## Figure Slots

Create your images under `docs/images/` and keep the following filenames so the README remains consistent.

### 1. Framework or UI overview

Reserved path: `docs/images/framework-overview.png`

Suggested content:

- Streamlit interface screenshot
- computation graph visualization
- intermediate activation map screenshot

Markdown to enable after adding the figure:

```md
![Framework overview](docs/images/framework-overview.png)
```

### 2. SST-2 training curves

Reserved path: `docs/images/sst2-training-curves.png`

Suggested content:

- train loss vs epoch
- train accuracy vs epoch
- validation accuracy vs epoch

Markdown to enable after adding the figure:

```md
![SST-2 training curves](docs/images/sst2-training-curves.png)
```

### 3. MNIST training curves

Reserved path: `docs/images/mnist-training-curves.png`

Suggested content:

- train loss vs epoch
- validation accuracy vs epoch

Markdown to enable after adding the figure:

```md
![MNIST training curves](docs/images/mnist-training-curves.png)
```

### 4. Final metrics summary

Reserved path: `docs/images/final-metrics-summary.png`

Suggested content:

- best SST-2 validation accuracy
- best MNIST validation accuracy
- MiniTorch vs modern pipeline comparison table

Markdown to enable after adding the figure:

```md
![Final metrics summary](docs/images/final-metrics-summary.png)
```

## Project Value

This project is useful from both a learning and engineering perspective:

- It demonstrates understanding of how deep learning frameworks implement autodiff, tensor semantics, and neural network operators.
- It shows the ability to connect low-level framework code with full training pipelines on real datasets.
- It also reflects practical engineering work, including environment setup, data handling, training reproducibility, and debugging across different runtime stacks.

## Resume-Friendly Summary

You can summarize this project on a resume as:

> Built and extended a MiniTorch-based deep learning systems project, implementing autodiff, tensor operators, and CNN modules, then validated the framework on SST-2 sentiment classification and MNIST image classification.

## Notes

- The original `project/` scripts are coursework-oriented and intentionally keep the MiniTorch style.
- The `scripts/` directory contains modern training entrypoints for easier reproduction and demonstration.
- If dataset downloads are unstable, prefer setting `HF_HOME` and `EMBEDDINGS_ROOT` to a persistent local cache directory before running training.
