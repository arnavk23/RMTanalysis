## Informer + WeightWatcher: Encoder Layer Spectral Analysis

This project integrates the [WeightWatcher](https://github.com/CalculatedContent/WeightWatcher) framework with the [Informer (2020)](https://github.com/zhouhaoyi/Informer2020) time-series transformer model. The goal is to extract encoder layer weights and analyze their singular value spectra (SVS), providing insight into the spectral properties and training quality of each layer.

### Structure

- **Informer2020/**: Original Informer implementation (model code, scripts, etc.)
- **notebooks/**:
  - `Informer_WeightWatcher_Analysis.ipynb`: notebook for encoder traversal, SVS extraction, and visualization.
  - `Singular_Value_analysis.ipynb`: Main notebook for SVD and singular value spectrum analysis.
  - `informer_weightwatcher_details.csv`, `informer_encoder_details.csv`: WeightWatcher analysis results (all layers and encoder layers).
- **checkpoints/**: Model checkpoints (random weights).

## Approach

1. **Environment Setup**
	- Cloned `Informer2020`, installed `WeightWatcher` and supporting libraries (`PyTorch`, `pandas`, `matplotlib`).

2. **Model Inspection & Weight Extraction**
	- Loaded the Informer model and traversed all encoder submodules (attention, FFN, conv layers).
	- Extracted weight matrices for each relevant encoder sublayer.

3. **WeightWatcher Analysis**
	- Ran WeightWatcher on the full model and filtered results for encoder layers.
	- Saved detailed metrics (e.g., `alpha`, `D`, `sigma`, `sv_min`, `sv_max`, `warnings`) to CSV files.

4. **Singular Value Spectrum (SVS) Computation**
	- Computed SVD for each encoder sublayer (`Linear`, `Conv1d`, `Embedding`).
	- Visualized SVS distributions and compared under-trained vs. well-trained layers.

## Main Findings

- **Encoder Layer Spectra**: Most layers show a heavy-tailed SVS, but some are flagged as "under-trained" by WeightWatcher (see CSVs and notebook plots).

- **Under-trained layers**: Identified by WeightWatcher warnings, low alpha, and narrow SVS. These may require more training or architectural adjustment.

- **Well-trained layers**: Show broader SVS, higher D, and no warnings.

- **WeightWatcher Metrics**: Layers with low `alpha` or `D` are often flagged as under-trained.

- **Interpretation**: Under-trained layers typically have a narrow SVS and low `alpha`, indicating limited learning or capacity usage. Well-trained layers show broader SVS and more favorable metrics.

## How to Reproduce

1. **Clone this repository**
2. **Install dependencies**:
	```bash
	pip install -r requirements.txt
	pip install weightwatcher
	```
3. **Run the analysis notebooks** in `notebooks/` for full code and results.
4. **(Optional)**: Run scripts such as `analyze_encoder_weightwatcher.py` for automated analysis and CSV generation.

## Key Results

See the CSVs and notebooks for full tables and plots. Example (abridged):

| Layer | Alpha | D | Sigma | sv_min | sv_max | Warning |
|-------|-------|---|-------|--------|--------|---------|
| encoder.attn_layers.0.attention.query_projection | 10.13 | 0.11 | 1.72 | 0.00015 | 1.15 | under-trained |
| encoder.attn_layers.0.attention.key_projection   | 9.87  | 0.13 | 1.62 | 0.00009 | 1.13 | under-trained |
| encoder.attn_layers.1.attention.value_projection | 20.70 | 0.12 | 5.68 | 0.00032 | 1.13 | under-trained |

## Contact

For questions or further discussion, please contact me on mail : arnavkapoor23@iiserb.ac.in and the repository is licensed under MIT License.

---