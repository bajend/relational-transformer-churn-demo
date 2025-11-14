import gradio as gr
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

# Hardcoded approx stats (from paper; in real, compute from your data or RelBench train split)
NUM_MEAN, NUM_STD = 0.0, 1.0  # For prices/ratings
DT_MEAN, DT_STD = 1440000000.0, 100000000.0  # Unix timestamps ~2015

# Text embedder
embedder = SentenceTransformer('all-MiniLM-L12-v2')

# Load model
ckpt_path = Path(hf_hub_download(
    repo_id="rishabh-ranjan/relational-transformer",
    filename="contd-pretrain_rel-amazon_user-churn.pt",
    local_dir="./ckpts"
))
model = RelationalTransformer(
    num_blocks=12, d_model=256, d_text=384, num_heads=8, d_ff=1024
)
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model = model.to(torch.bfloat16)
model.eval()

# Default synthetic data (CSV strings)
default_customer_csv = """customer_id,name
A1,Alice
A2,Bob
A3,Charlie
"""
default_product_csv = """asin,title,description,price
P1,Book1,Adventure novel,10.99
P2,Book2,Sci-fi thriller,15.49
P3,Book3,Mystery,12.99
P4,Book4,Romance,9.99
P5,Book5,History,20.00
"""
default_review_csv = """customer_id,asin,rating,review_text,timestamp
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

def parse_csv_to_df(csv_str, schema):
    from io import StringIO
    return pl.read_csv(StringIO(csv_str), schema=schema)

def build_context(seed_customer_id, pred_timestamp_str, customer_df, product_df, review_df):
    pred_ts = datetime.strptime(pred_timestamp_str, '%Y-%m-%d').timestamp()
    
    # Task row: [customer_id, timestamp, churn (masked)]
    task_df = pl.DataFrame({'customer_id': [seed_customer_id], 'timestamp': [pred_ts], 'churn': [None]})
    
    # BFS: Start from task -> customer -> reviews -> products
    customer_row = customer_df.filter(pl.col('customer_id') == seed_customer_id)
    reviews = review_df.filter(pl.col('customer_id') == seed_customer_id)
    products = product_df.join(reviews.select('asin'), on='asin', how='inner')
    
    # Collect cells (flatten to list of (table, col, value, row_idx))
    cells = []
    # Add task cells (mask churn)
    cells.append(('user-churn', 'customer_id', seed_customer_id, 0))
    cells.append(('user-churn', 'timestamp', pred_ts, 0))
    cells.append(('user-churn', 'churn', None, 0))  # Masked target
    
    # Add customer cells
    for col in customer_row.columns:
        if col != 'customer_id':
            cells.append(('customer', col, customer_row[col][0], 0))
    
    # Add reviews and linked products (limit to max 256 for demo)
    for i, row in enumerate(reviews.iter_rows(named=True)):
        for col in reviews.columns:
            if col not in ['customer_id', 'asin']:
                val = row[col]
                if col == 'timestamp' and isinstance(val, str):
                    val = datetime.strptime(val, '%Y-%m-%d').timestamp()
                cells.append(('review', col, val, i))
        # Linked product
        prod_row = products.filter(pl.col('asin') == row['asin'])
        for col in prod_row.columns:
            if col != 'asin':
                cells.append(('product', col, prod_row[col][0], i))
    
    return cells

def prepare_batch(cells, seq_len=1024):
    # Simplified tensor construction (mimic rt/data.py)
    # Embed texts, normalize nums/dts, set sem_types, masks, etc.
    dtype = torch.bfloat16
    node_idxs = torch.arange(len(cells))  # Mock
    sem_types = torch.zeros(len(cells), dtype=torch.long)  # 0=number,1=text,2=datetime,3=boolean
    masks = torch.zeros(len(cells), dtype=torch.bool)  # Only mask the target
    is_targets = torch.zeros(len(cells), dtype=torch.bool)
    is_task_nodes = torch.ones(len(cells), dtype=torch.bool)  # Assume all task-related
    is_padding = torch.zeros(len(cells), dtype=torch.bool)
    table_name_idxs = torch.zeros(len(cells), dtype=torch.long)  # Mock indices
    col_name_idxs = torch.zeros(len(cells), dtype=torch.long)
    
    table_to_idx = {}
    col_to_idx = {}
    
    for i, (table, col, val, row_idx) in enumerate(cells):
        table_idx = table_to_idx.setdefault(table, len(table_to_idx))
        col_key = f"{table}.{col}"
        col_idx = col_to_idx.setdefault(col_key, len(col_to_idx))
        
        table_name_idxs[i] = table_idx
        col_name_idxs[i] = col_idx
    class_value_idxs = torch.full((len(cells),), -1)
    f2p_nbr_idxs = torch.full((len(cells), 5), -1)  # Mock links
    number_values = torch.zeros(len(cells), 1, dtype=dtype)
    datetime_values = torch.zeros(len(cells), 1, dtype=dtype)
    boolean_values = torch.zeros(len(cells), 1, dtype=dtype)
    text_values = torch.zeros(len(cells), 384, dtype=dtype)  # d_text=384
    col_name_values = torch.zeros(len(cells), 384, dtype=dtype)
    
    for i, (table, col, val, row_idx) in enumerate(cells):
        col_phrase = f"{col} of {table}"
        col_emb = embedder.encode([col_phrase])[0]
        col_name_values[i] = torch.from_numpy(col_emb).to(dtype)
        
        if col == 'churn':
            is_targets[i] = True
            sem_types[i] = 3  # Boolean
            boolean_values[i] = torch.tensor(0.0, dtype=dtype)
            masks[i] = True  # Mask the target for prediction
        elif isinstance(val, float) and col == 'timestamp':
            sem_types[i] = 2  # Datetime
            datetime_values[i] = torch.tensor((val - DT_MEAN) / DT_STD, dtype=dtype)
        elif isinstance(val, (int, float)):
            sem_types[i] = 0  # Number
            number_values[i] = torch.tensor((val - NUM_MEAN) / NUM_STD if val is not None else 0.0, dtype=dtype)
        elif isinstance(val, str):
            sem_types[i] = 1  # Text
            text_emb = embedder.encode([val])[0]
            text_values[i] = torch.from_numpy(text_emb).to(dtype)
    
    text_to_idx = {}  # Mock
    column_index = {}  # Mock
    
    for i, (table, col, val, row_idx) in enumerate(cells):
        col_phrase = f"{col} of {table}"
        col_emb = embedder.encode([col_phrase])[0]
        col_name_values[i] = torch.from_numpy(col_emb).to(dtype)
        
        if col == 'churn':
            is_targets[i] = True
            sem_types[i] = 0  # Boolean
            boolean_values[i] = torch.tensor(0.0, dtype=dtype)
        elif isinstance(val, float) and col == 'timestamp':
            sem_types[i] = 2  # DateTime
            datetime_values[i] = torch.tensor((val - DT_MEAN) / DT_STD, dtype=dtype)
        elif isinstance(val, (int, float)):
            sem_types[i] = 1  # Number
            number_values[i] = torch.tensor((val - NUM_MEAN) / NUM_STD if val is not None else 0.0, dtype=dtype)
        elif isinstance(val, str):
            sem_types[i] = 3  # Text
            text_emb = embedder.encode([val])[0]
            text_values[i] = torch.from_numpy(text_emb).to(dtype)
            # Mock class_value_idx if needed
        # Handle potential None values not churn
        if val is None and col != 'churn':
            # Skip or handle, for demo assume no
            pass
        # Mock f2p links (e.g., review to customer/product)
        if table == 'review':
            f2p_nbr_idxs[i, 0] = 0  # Link to customer row 0
    
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
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    return batch

def predict_churn(customer_csv, product_csv, review_csv, customer_id, pred_timestamp):
    # Schemas (datatypes)
    customer_schema = {'customer_id': pl.Utf8, 'name': pl.Utf8}
    product_schema = {'asin': pl.Utf8, 'title': pl.Utf8, 'description': pl.Utf8, 'price': pl.Float64}
    review_schema = {'customer_id': pl.Utf8, 'asin': pl.Utf8, 'rating': pl.Int32, 'review_text': pl.Utf8, 'timestamp': pl.Utf8}
    
    customer_df = parse_csv_to_df(customer_csv, customer_schema)
    product_df = parse_csv_to_df(product_csv, product_schema)
    review_df = parse_csv_to_df(review_csv, review_schema)
    
    pred_ts = datetime.fromisoformat(pred_timestamp).timestamp()
    
    # Convert timestamp in review_df to unix seconds
    review_df = review_df.with_columns(
        pl.col('timestamp').str.strptime(pl.Datetime, '%Y-%m-%d').dt.timestamp() / 1e9
    )
    
    # Filter reviews before pred_ts
    review_df = review_df.filter(pl.col('timestamp') < pred_ts)
    
    cells = build_context(customer_id, pred_timestamp, customer_df, product_df, review_df)
    batch = prepare_batch(cells)
    
    with torch.no_grad():
        _, yhat_dict = model(batch)
        churn_logit = yhat_dict['boolean'][batch['is_targets']].item()
        prob = torch.sigmoid(torch.tensor(churn_logit)).item()
    
    print(f"Debug: Customer {customer_id}, Timestamp {pred_timestamp}, Churn logit: {churn_logit:.4f}, Prob: {prob:.4f}")
    print(f"Debug: Number of cells: {len(cells)}")
    
    return f"Churn Probability: {prob:.2%} (>50% means likely churn: no reviews in next 3 months)"

with gr.Blocks() as demo:
    gr.Markdown("# Relational Transformer Demo: Predict User Churn on rel-amazon")
    with gr.Row():
        customer_input = gr.TextArea(value=default_customer_csv, label="Customer Table (CSV)")
        product_input = gr.TextArea(value=default_product_csv, label="Product Table (CSV)")
    review_input = gr.TextArea(value=default_review_csv, label="Review Table (CSV)")
    customer_id = gr.Textbox(value="A1", label="Customer ID to Predict")
    pred_timestamp = gr.Textbox(value="2015-10-01", label="Prediction Timestamp (YYYY-MM-DD)")
    predict_btn = gr.Button("Predict Churn")
    output = gr.Textbox(label="Prediction", lines=2)
    
    predict_btn.click(
        predict_churn,
        inputs=[customer_input, product_input, review_input, customer_id, pred_timestamp],
        outputs=output
    )

demo.launch()