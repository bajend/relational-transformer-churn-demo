#!/bin/bash

# Script to set up and run the Relational Transformer Churn Prediction Demo
# This script executes all the steps from cloning the repo to launching the UI

echo "Step 1: Cloning the Relational Transformer repository..."
git clone https://github.com/snap-stanford/relational-transformer ./relational-transformer

echo "Step 2: Installing Python dependencies..."
pip install torch==2.6.0 sentence_transformers wandb einops strictfire maturin ml_dtypes orjson polars scikit-learn<1.6.0 matplotlib gradio huggingface_hub

echo "Step 3: Installing compatible torchvision version..."
pip install torchvision==0.21.0

echo "Step 4: Downloading the pretrained model checkpoint..."
python -c "
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(
    repo_id='rishabh-ranjan/relational-transformer',
    filename='contd-pretrain_rel-amazon_user-churn.pt',
    local_dir='./ckpts'
)
print(f'Checkpoint downloaded to: {ckpt_path}')
"

echo "Step 5: Launching the demo UI..."
cd /teamspace/studios/this_studio
python demo.py