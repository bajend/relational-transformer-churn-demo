# Relational Transformer Churn Prediction Demo

A playground UI for demonstrating churn prediction using the Relational Transformer (RT) model on synthetic `rel-amazon` dataset data.

## Overview

This demo showcases the Relational Transformer model, a powerful architecture for relational reasoning tasks. It predicts user churn (whether a customer will stop writing reviews) based on their review history and product interactions.

The demo uses:
- **Model**: Relational Transformer (12 blocks, d_model=256, d_text=384, 8 heads, d_ff=1024)
- **Task**: Binary classification for user churn prediction
- **Data**: Synthetic data mimicking the `rel-amazon` schema (customers, products, reviews)
- **Inference**: Zero-shot prediction using a pretrained checkpoint

## Features

- Interactive Gradio UI for editing tables and predicting churn
- Synthetic data generation for customers, products, and reviews
- Real-time churn probability prediction
- Context-aware reasoning using relational structures

## Setup and Installation

### Prerequisites
- Python 3.8+
- Git

### Quick Start

1. **Clone this repository**:
   ```bash
   git clone https://github.com/bajend/relational-transformer-churn-demo.git
   cd relational-transformer-churn-demo
   ```

2. **Run the setup script**:
   ```bash
   chmod +x setup_and_run_demo.sh
   ./setup_and_run_demo.sh
   ```

   This script will:
   - Clone the Relational Transformer repository
   - Install all required dependencies
   - Download the pretrained model checkpoint
   - Launch the demo UI

### Manual Setup

If you prefer to set up manually:

1. **Clone dependencies**:
   ```bash
   git clone https://github.com/snap-stanford/relational-transformer ./relational-transformer
   ```

2. **Install dependencies**:
   ```bash
   pip install torch==2.6.0 sentence_transformers wandb einops strictfire maturin ml_dtypes orjson polars scikit-learn<1.6.0 matplotlib gradio huggingface_hub
   pip install torchvision==0.21.0
   ```

3. **Download checkpoint**:
   ```python
   from huggingface_hub import hf_hub_download
   ckpt_path = hf_hub_download(
       repo_id="rishabh-ranjan/relational-transformer",
       filename="contd-pretrain_rel-amazon_user-churn.pt",
       local_dir="./ckpts"
   )
   ```

4. **Run the demo**:
   ```bash
   python demo.py
   ```

## Usage

1. Open the Gradio UI at `http://127.0.0.1:7860`
2. Edit the CSV data in the text areas (customers, products, reviews)
3. Select a customer ID and prediction timestamp
4. Click "Predict Churn" to get the probability

The model analyzes the relational context (customer reviews, linked products) to predict whether the user will churn (no reviews in the next 3 months).

## Architecture

The Relational Transformer processes:
- **Customer table**: User information
- **Product table**: Item details with titles, descriptions, prices
- **Review table**: User-item interactions with ratings, text, timestamps
- **Task table**: Churn prediction target

The model uses:
- Multi-head attention for relational reasoning
- Text embeddings for semantic understanding
- Normalized numerical and temporal features
- BFS sampling for context construction

## Model Details

- **Pretrained on**: `rel-amazon` dataset (continued pretraining)
- **Task**: User churn prediction
- **Metric**: AUROC
- **Architecture**: 12 transformer blocks, 256D model, 384D text embeddings

## Screenshots

./Screenshot 2025-11-13 072022.jpg

## Citation

If you use this demo, please cite the original Relational Transformer paper:

```bibtex
@article{relational_transformer,
  title={Relational Transformer},
  author={Your Author},
  journal={arXiv preprint arXiv:2510.06377},
  year={2025}
}
```

## License

This project is for educational and demonstration purposes. Check the licenses of the dependencies used.
