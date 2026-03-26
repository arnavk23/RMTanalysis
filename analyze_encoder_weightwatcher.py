import torch
import weightwatcher as ww
import pandas as pd
from pathlib import Path
import sys
import os

# --- Setup paths ---
INFORMER_PATH = Path("./Informer2020")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Informer2020')))

from models.model import Informer

# --- Instantiate model (match your config!) ---
model = Informer(
    enc_in=7, dec_in=7, c_out=7,  # adjust for your dataset
    seq_len=96, label_len=48, out_len=24,
    factor=5, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
    dropout=0.05, attn='prob', embed='fixed', freq='h', activation='gelu',
    output_attention=False, distil=True, mix=True, device='cpu'
)


# --- Load checkpoint if available, else save a random one ---
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)
checkpoint = checkpoint_dir / "random_informer.pth"
if checkpoint.exists():
    state = torch.load(checkpoint, map_location='cpu')
    if 'model' in state:
        model.load_state_dict(state['model'])
    elif 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    print(f"Checkpoint loaded: {checkpoint}")
else:
    print("No checkpoint found. Saving randomly initialized model.")
    torch.save(model.state_dict(), checkpoint)
    print(f"Random checkpoint saved to: {checkpoint}")

model.eval()


# --- Analyze the full model (WeightWatcher will find all Linear/Conv layers) ---
watcher = ww.WeightWatcher(model=model)
details = watcher.analyze(plot=True, savefig="esd_encoder_layers.png")
details.to_csv("informer_weightwatcher_details.csv", index=False)

# --- Filter for encoder layers only (optional, for reporting) ---
if 'layer_type' in details.columns:
    encoder_details = details[details['layer_type'].str.contains('encoder', case=False, na=False)]
else:
    encoder_details = details
encoder_details.to_csv("informer_encoder_details.csv", index=False)

# Print available columns and a preview for debugging
print("Encoder details columns:", encoder_details.columns.tolist())
print(encoder_details.head())

# Try to print key columns if they exist
cols = [c for c in ['layer_id', 'layer', 'alpha', 'log_norm', 'log_alpha', 'summary'] if c in encoder_details.columns]
if cols:
    print(encoder_details[cols])
else:
    print("Some expected columns not found in encoder_details.")

# --- (Optional) Extract and plot singular values for each encoder submodule ---
def collect_encoder_weights(module, prefix=""):
    for name, child in module.named_children():
        if isinstance(child, (torch.nn.Linear, torch.nn.Conv1d)):
            w = child.weight.detach().cpu()
            if w.ndim > 2:
                w = w.reshape(w.shape[0], -1)
            s = torch.linalg.svdvals(w).numpy()
            print(f"{prefix}.{name}: singular values shape {s.shape}, max {s.max():.4f}, min {s.min():.4f}")
        collect_encoder_weights(child, f"{prefix}.{name}")

print("Encoder layer singular values:")
collect_encoder_weights(model.encoder, "encoder")
