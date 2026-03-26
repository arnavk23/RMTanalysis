# Script to extract Informer encoder weights and analyze with WeightWatcher
import torch
import weightwatcher as ww
import matplotlib.pyplot as plt
import os

# --- USER: Set this path to your Informer repo and model checkpoint ---
INFORMER_REPO_PATH = './Informer2020'
# Place your Informer checkpoint in the 'checkpoints' directory and set the filename below
CHECKPOINT_PATH = None  # Update this filename as needed

# --- Import Informer model ---
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Informer2020')))
from models.model import Informer

# --- Load Informer model (edit args as needed) ---
model = Informer(
    enc_in=1, dec_in=1, c_out=1, seq_len=96, label_len=48, out_len=24,
    factor=5, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
    dropout=0.05, attn='prob', embed='fixed', freq='h', activation='gelu',
    output_attention=False, distil=True, mix=True, device='cpu'
)
if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
    state = torch.load(CHECKPOINT_PATH, map_location='cpu')
    # Some checkpoints may have 'model' or 'state_dict' keys
    if 'model' in state:
        model.load_state_dict(state['model'])
    elif 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
else:
    print(f"No checkpoint loaded at {CHECKPOINT_PATH}. Using randomly initialized model.")

# --- Extract encoder weights ---

encoder = model.encoder
layer_weights = []
for i, layer in enumerate(encoder.attn_layers):
    print(f"\nEncoder Layer {i}:")
    for name, param in layer.named_parameters():
        if param.dim() >= 2:
            print(f"  {name}: {param.shape}")
            layer_weights.append((f"encoder_layer_{i}_{name}", param.detach().numpy()))

# --- Analyze with WeightWatcher ---

import torch.nn as nn
for name, weights in layer_weights:
    print(f"\nAnalyzing {name}...")
    # Flatten conv weights to 2D if needed
    if weights.ndim > 2:
        w = weights.reshape(weights.shape[0], -1)
    else:
        w = weights
    # Create a dummy nn.Module with a single Linear layer for WeightWatcher
    class Dummy(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.linear = nn.Linear(w.shape[1], w.shape[0], bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor(w))
        def forward(self, x):
            return self.linear(x)
    dummy = Dummy(w)
    watcher = ww.WeightWatcher(dummy)
    details = watcher.analyze(plot=True, savefig=f"{name}_svd.png")
    print(details)
    plt.close('all')

print("\nAnalysis complete. Singular value plots saved as PNG files.")
