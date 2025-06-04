# Single-Molecule Trajectory Analysis

This code performs point-by-point prediction of adsorption/desorption (bulk diffusion) events from single-molecule trajectories, serving as supplementary material for the paper **"In-situ Probing of Polymer Residence on the Nanoparticle Surface by Combining Deep Learning and Single-Molecule Fluorescence Tracking"**.

## Repository Structure

### `experiment_code/` 
Contains all code, test sets, trained models, and processed results used in the paper for reproducibility.  
**Note:** Due to file size constraints, only a test set of 2,000 polymer trajectories is included (representing the 40nm nanoparticle system with adsorption/bulk diffusion coefficient ratio D*=0.5).

### `demo/` 
Provides a complete workflow demonstration including:
- Data processing (`data_preprocess.py`)
- Model training (`train.py`)
- Testing procedures (`test.py`)
- Post-processing scripts (integrated in `test.py`)

The raw data for this demo is located in `simorig0/`. Due to GitHub file count limitations, only 100 trajectory files are included.

## Getting Started
1. Navigate to the `demo/` directory
2. Run `demo.ipynb` using Jupyter Notebook:
```bash
jupyter notebook demo.ipynb
