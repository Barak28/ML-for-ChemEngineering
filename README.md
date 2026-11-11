# ML for Chemical Engineering

This repository contains a structured set of Jupyter notebooks covering machine learning methods for chemical engineering applications.

## Repository structure

The lessons are organized by topic:

- `lesson_01_learning_data_scarcity`
- `lesson_02_optimization_regression_classification`
- `lesson_03_dimensionality_reduction_latent_variables`
- `lesson_04_svm_kernels`
- `lesson_05_trees_ensembles`
- `lesson_06_unsupervised_anomaly_detection`
- `lesson_07_gaussian_processes_active_learning`
- `lesson_08_neural_networks_universal_approximation`
- `lesson_09_optimization_dynamics_regularization`
- `lesson_10_cnn_spatial_data`
- `lesson_11_recurrent_time_series`
- `lesson_12_self_attention_transformers`
- `lesson_13_xai_regulatory_compliance`
- `lesson_14_physics_informed_neural_networks`

## Quick start

1. Create a Python environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open any lesson notebook and run cells in order.

## Recommended notebook practices

To improve the quality and readability of the notebooks across the repository, the following conventions are recommended:

- Keep imports at the top of each notebook.
- Define a single random seed early for reproducibility.
- Split data into train/validation/test sets whenever appropriate.
- Use descriptive function names and comments.
- Wrap repeated logic in utilities instead of duplicating code.
- Place plotting code in dedicated sections for easy reuse.
- Report evaluation metrics explicitly, not only plots.
- Document assumptions and limitations of each model.

## Shared utilities

A reusable helper module is included at the repository root:

- `notebook_utils.py`

It provides common utilities for:

- seeding experiments,
- plotting configuration,
- regression/classification evaluation,
- consistent notebook setup.

## Future improvements

The next logical upgrades for this repository are:

- convert repeated notebook logic into `src/` modules,
- add CI checks for notebook execution,
- standardize naming across lessons,
- include versioned environment files,
- add data downloads and cache management for reproducible runs.

## License

This repository is for educational purposes and is intended to support learning and experimentation in machine learning for chemical engineering.
