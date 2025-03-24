# SLUE Speech Language Understanding Project

This repository contains code for processing and analyzing the SLUE dataset.

## Project Structure

- `data/`: Contains raw SLUE dataset and processed features
- `notebooks/`: Jupyter notebooks for exploratory data analysis and feature extraction
- `src/`: Source code for data loading, feature extraction, and model training
- `models/`: Saved trained models
- `results/`: Evaluation metrics and plots

## Getting Started

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Prepare the data:
   - Place raw SLUE dataset in `data/raw/`
   - Run feature extraction to generate processed features in `data/processed/`

3. Train models:
   ```
   python src/train.py
   ```

## License

[Your License Here] 