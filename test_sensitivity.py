#!/usr/bin/env python3
"""
Test script for Relational Transformer Churn Prediction Model Sensitivity
This script tests how the model's predictions change with different input data configurations.
Run this to verify the model is sensitive to input variations.
"""

import torch
import polars as pl
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
from pathlib import Path
import sys
sys.path.append('./relational-transformer')
from rt.model import RelationalTransformer

# Hardcoded stats (same as demo)
NUM_MEAN, NUM_STD = 0.0, 1.0
DT_MEAN, DT_STD = 1440000000.0, 100000000.0

# Load embedder
embedder = SentenceTransformer('all-MiniLM-L12-v2')

# Load model
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device('cpu')  # Force CPU for testing
print(f"Using device: {device}")

ckpt_path = Path(hf_hub_download(
    repo_id="rishabh-ranjan/relational-transformer",
    filename="contd-pretrain_rel-amazon_user-churn.pt",
    local_dir="./ckpts"
))
model = RelationalTransformer(
    num_blocks=12, d_model=256, d_text=384, num_heads=8, d_ff=1024
)
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
model = model.to(dtype=torch.float32)  # Ensure model weights are float32
model.eval()
model.to(device)

def parse_csv_to_df(csv_str, schema):
    from io import StringIO
    return pl.read_csv(StringIO(csv_str), schema=schema)

def build_context(seed_customer_id, pred_timestamp_str, customer_df, product_df, review_df):
    pred_ts = datetime.strptime(pred_timestamp_str, '%Y-%m-%d').timestamp()

    # Task row
    task_df = pl.DataFrame({'customer_id': [seed_customer_id], 'timestamp': [pred_ts], 'churn': [None]})

    # BFS: customer -> reviews -> products
    customer_row = customer_df.filter(pl.col('customer_id') == seed_customer_id)
    reviews = review_df.filter(pl.col('customer_id') == seed_customer_id)
    products = product_df.join(reviews.select('asin'), on='asin', how='inner')

    cells = []
    # Task cells
    cells.append(('user-churn', 'customer_id', seed_customer_id, 0))
    cells.append(('user-churn', 'timestamp', pred_ts, 0))
    cells.append(('user-churn', 'churn', None, 0))

    # Customer cells
    for col in customer_row.columns:
        if col != 'customer_id':
            cells.append(('customer', col, customer_row[col][0], 0))

    # Reviews and products
    for i, row in enumerate(reviews.iter_rows(named=True)):
        for col in reviews.columns:
            if col not in ['customer_id', 'asin']:
                cells.append(('review', col, row[col], i))
        prod_row = products.filter(pl.col('asin') == row['asin'])
        for col in prod_row.columns:
            if col != 'asin':
                cells.append(('product', col, prod_row[col][0], i))

    return cells

def prepare_batch(cells, seq_len=1024):
    dtype = torch.float32  # Change to float32 to match model weights
    node_idxs = torch.arange(len(cells))
    sem_types = torch.zeros(len(cells), dtype=torch.long)
    masks = torch.zeros(len(cells), dtype=torch.bool)
    is_targets = torch.zeros(len(cells), dtype=torch.bool)
    is_task_nodes = torch.ones(len(cells), dtype=torch.bool)
    is_padding = torch.zeros(len(cells), dtype=torch.bool)
    table_name_idxs = torch.zeros(len(cells), dtype=torch.long)
    col_name_idxs = torch.zeros(len(cells), dtype=torch.long)
    class_value_idxs = torch.full((len(cells),), -1, dtype=torch.long)
    f2p_nbr_idxs = torch.full((len(cells), 5), -1, dtype=torch.long)
    number_values = torch.zeros(len(cells), 1, dtype=dtype)
    datetime_values = torch.zeros(len(cells), 1, dtype=dtype)
    boolean_values = torch.zeros(len(cells), 1, dtype=dtype)
    text_values = torch.zeros(len(cells), 384, dtype=dtype)
    col_name_values = torch.zeros(len(cells), 384, dtype=dtype)
    
    table_to_idx = {}
    col_to_idx = {}
    
    # --- DEBUG FLAG ---
    text_debug_printed = False
    
    for i, (table, col, val, row_idx) in enumerate(cells):
        table_idx = table_to_idx.setdefault(table, len(table_to_idx))
        col_key = f"{table}.{col}"
        col_idx = col_to_idx.setdefault(col_key, len(col_to_idx))
        
        table_name_idxs[i] = table_idx
        col_name_idxs[i] = col_idx
        
        col_phrase = f"{col} of {table}"
        col_emb = embedder.encode([col_phrase])[0]
        col_name_values[i] = torch.from_numpy(col_emb).to(dtype)
        
        if col == 'churn':
            is_targets[i] = True
            sem_types[i] = 3  # Boolean
            boolean_values[i] = torch.tensor(0.0, dtype=dtype)
            masks[i] = True
        elif isinstance(val, float) and col == 'timestamp':
            sem_types[i] = 2  # Datetime
            datetime_values[i] = torch.tensor((val - DT_MEAN) / DT_STD, dtype=dtype)
        elif isinstance(val, (int, float)):
            sem_types[i] = 0  # Number
            number_values[i] = torch.tensor((val - NUM_MEAN) / NUM_STD if val is not None else 0.0, dtype=dtype)
        elif isinstance(val, str):
            sem_types[i] = 1  # Text
            
            # --- START DEBUG PRINT ---
            # Encode the text
            text_emb_np = embedder.encode([val])[0]
            text_values[i] = torch.from_numpy(text_emb_np).to(dtype)
            
            # Print details for the first 'review_text' we find
            if not text_debug_printed and table == 'review' and col == 'review_text':
                print("\n--- Embedding Debug Start ---")
                print(f"  [Debug] Encoding text for: {table}.{col}")
                print(f"  [Debug] Raw Text: '{val}'")
                print(f"  [Debug] Embedding Shape: {text_emb_np.shape}")
                print(f"  [Debug] Embedding (first 5 values): {text_emb_np[:5]}")
                print("--- Embedding Debug End ---\n")
                text_debug_printed = True
            # --- END DEBUG PRINT ---
    
    batch = {
        'node_idxs': node_idxs.unsqueeze(0),
        'sem_types': sem_types.unsqueeze(0),
        'masks': masks.unsqueeze(0),
        'is_targets': is_targets.unsqueeze(0),
        'is_task_nodes': is_task_nodes.unsqueeze(0),
        'is_padding': is_padding.unsqueeze(0),
        'table_name_idxs': table_name_idxs.unsqueeze(0),
        'col_name_idxs': col_name_idxs.unsqueeze(0),
        'class_value_idxs': class_value_idxs.unsqueeze(0),
        'f2p_nbr_idxs': f2p_nbr_idxs.unsqueeze(0),
        'number_values': number_values.unsqueeze(0),
        'datetime_values': datetime_values.unsqueeze(0),
        'boolean_values': boolean_values.unsqueeze(0),
        'text_values': text_values.unsqueeze(0),
        'col_name_values': col_name_values.unsqueeze(0),
        'true_batch_size': 1
    }
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

def predict_churn(customer_csv, product_csv, review_csv, customer_id, pred_timestamp):
    customer_schema = {'customer_id': pl.Utf8, 'name': pl.Utf8}
    product_schema = {'asin': pl.Utf8, 'title': pl.Utf8, 'description': pl.Utf8, 'price': pl.Float64}
    review_schema = {'customer_id': pl.Utf8, 'asin': pl.Utf8, 'rating': pl.Int32, 'review_text': pl.Utf8, 'timestamp': pl.Utf8}

    customer_df = parse_csv_to_df(customer_csv, customer_schema)
    product_df = parse_csv_to_df(product_csv, product_schema)
    review_df = parse_csv_to_df(review_csv, review_schema)

    pred_ts = datetime.fromisoformat(pred_timestamp).timestamp()
    
    # --- THIS IS THE CORRECTED LINE ---
    review_df = review_df.with_columns(
        pl.col('timestamp').str.strptime(pl.Datetime, '%Y-%m-%d').dt.timestamp(time_unit='ms') / 1000.0
    )
    # ----------------------------------
    
    review_df = review_df.filter(pl.col('timestamp') < pred_ts)

    cells = build_context(customer_id, pred_timestamp, customer_df, product_df, review_df)
    batch = prepare_batch(cells)
    
    print(f"  Prepared batch with {len(cells)} cells")
    
    with torch.no_grad():
        print("  Running model forward...")
        _, yhat_dict = model(batch)
        print("  Model forward complete")
        churn_logit = yhat_dict['boolean'][batch['is_targets']].item()
        prob = torch.sigmoid(torch.tensor(churn_logit)).item()
    
    return prob, len(cells)

# Test scenarios
base_customer_csv = """customer_id,name
A1,Alice
A2,Bob
A3,Charlie
"""

base_product_csv = """asin,title,description,price
P1,Book1,Adventure novel,10.99
P2,Book2,Sci-fi thriller,15.49
P3,Book3,Mystery,12.99
P4,Book4,Romance,9.99
P5,Book5,History,20.00
"""

base_review_csv = """customer_id,asin,rating,review_text,timestamp
A1,P1,5,Great book!,2015-01-01
A1,P2,4,Enjoyed it.,2015-02-01
A2,P3,3,Okay.,2015-03-01
A2,P1,5,Love it!,2015-04-01
A3,P4,2,Not great.,2015-05-01
A1,P5,4,Informative.,2015-06-01
A2,P2,5,Awesome!,2015-07-01
A3,P1,1,Bad.,2015-08-01
A1,P3,4,Good read.,2015-09-01
A2,P4,5,Fantastic!,2015-09-15
"""

# --- THIS BLOCK IS NOW CORRECTED ---
test_scenarios = [
    ("Original A1", base_customer_csv, base_product_csv, base_review_csv, "A1", "2015-10-01"),
    ("A1 with lower ratings", base_customer_csv, base_product_csv,
     """customer_id,asin,rating,review_text,timestamp
A1,P1,2,Poor book!,2015-01-01
A1,P2,1,Hated it.,2015-02-01
A2,P3,3,Okay.,2015-03-01
A2,P1,5,Love it!,2015-04-01
A3,P4,2,Not great.,2015-05-01
A1,P5,2,Boring.,2015-06-01
A2,P2,5,Awesome!,2015-07-01
A3,P1,1,Bad.,2015-08-01
A1,P3,1,Terrible.,2015-09-01
A2,P4,5,Fantastic!,2015-09-15
""", "A1", "2015-10-01"),
    ("A1 with fewer reviews", base_customer_csv, base_product_csv,
     """customer_id,asin,rating,review_text,timestamp
A1,P1,5,Great book!,2015-01-01
A2,P3,3,Okay.,2015-03-01
A2,P1,5,Love it!,2015-04-01
A3,P4,2,Not great.,2015-05-01
A2,P2,5,Awesome!,2015-07-01
A3,P1,1,Bad.,2015-08-01
A2,P4,5,Fantastic!,2015-09-15
""", "A1", "2015-10-01"),
    ("A1 with older last review", base_customer_csv, base_product_csv,
     """customer_id,asin,rating,review_text,timestamp
A1,P1,5,Great book!,2015-01-01
A1,P2,4,Enjoyed it.,2015-02-01
A2,P3,3,Okay.,2015-03-01
A2,P1,5,Love it!,2015-04-01
A3,P4,2,Not great.,2015-05-01
A1,P5,4,Informative.,2015-03-01
A2,P2,5,Awesome!,2015-07-01
A3,P1,1,Bad.,2015-08-01
A2,P4,5,Fantastic!,2015-09-15
""", "A1", "2015-10-01"),
    ("A1 with negative text", base_customer_csv, base_product_csv,
     """customer_id,asin,rating,review_text,timestamp
A1,P1,5,Worst book ever!,2015-01-01
A1,P2,4,Hated every page.,2015-02-01
A2,P3,3,Okay.,2015-03-01
A2,P1,5,Love it!,2015-04-01
A3,P4,2,Not great.,2015-05-01
A1,P5,4,Complete waste.,2015-06-01
A2,P2,5,Awesome!,2015-07-01
A3,P1,1,Bad.,2015-08-01
A1,P3,4,Disappointing.,2015-09-01
A2,P4,5,Fantastic!,2015-09-15
""", "A1", "2015-10-01"),
]
# -----------------------------------


if __name__ == "__main__":
    print("Testing Relational Transformer Churn Prediction Model Sensitivity")
    print("=" * 60)
    
    results = {}
    
    try:
        # Iterate through all scenarios and run the actual model
        for scenario_name, cust_csv, prod_csv, rev_csv, cust_id, pred_ts in test_scenarios:
            print(f"\nRunning scenario: {scenario_name}")
            
            # Call the actual prediction function
            prob, num_cells = predict_churn(
                cust_csv, prod_csv, rev_csv, cust_id, pred_ts
            )
            
            results[scenario_name] = prob
            
            print(f"  Churn Probability: {prob:.2%}")
            print(f"  Number of context cells: {num_cells}")

    except Exception as e:
        print(f"\nError during model inference: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Sensitivity Test Summary:")
    
    if not results:
        print("No scenarios were successfully run.")
    else:
        # Print a summary of all results
        base_prob = results.get("Original A1")
        print(f"  {'Scenario':<25} | {'Probability':<12} | {'Change from Base':<15}")
        print(f"  {'-'*25:<25} | {'-'*12:<12} | {'-'*15:<15}")
        
        for name, prob in results.items():
            change_str = ""
            if base_prob is not None and name != "Original A1":
                change = prob - base_prob
                change_str = f"{change:+.2%}"
            print(f"  {name:<25} | {prob:<12.2%} | {change_str:<15}")

    print("\nSensitivity test complete. Different inputs should show varying probabilities!")