# Neural Network Inverse Optimization

Project scaffold for generating process data, training and validating a neural
network model, performing inverse optimization, comparing optimizers, and
running closed-loop simulations.

Run `notebooks/01_generate_data.ipynb` with a Python kernel from the `notebooks/`
directory to generate `data/synthetic_process_data.csv` and inspect time-series,
output-distribution, and correlation plots. The notebook requires NumPy, pandas,
SciPy, Matplotlib, and seaborn; its `generate_process_data` function returns a
reproducible DataFrame with 8,000 sequential observations by default.

Run `notebooks/02_train_model.ipynb` next from `notebooks/` with PyTorch and
scikit-learn installed. It trains a 10-32-32-16-2 SiLU network with chronological
70/15/15 train/validation/test splits and train-only input/output scaling.
The notebook saves `process_model.pt`, `input_scaler.pkl`, and `output_scaler.pkl`
in `models/`, plus `training_metrics.csv`, `parity_plot.png`,
`residual_plots.png`, and `training_curves.png` in `results/`. The model
checkpoint contains the `state_dict`, column ordering, hidden sizes, and best
epoch; only load pickle scalers from trusted sources.
