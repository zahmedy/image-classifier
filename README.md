# Image Classifier (CIFAR-10)

Clean, minimal PyTorch project that trains a small CNN on CIFAR-10 and visualizes predictions. Great as a starter template or portfolio snippet.

## Project layout
```
image_classifier/
├─ README.md
├─ requirements.txt          # Python dependencies
├─ data/                     # Downloaded CIFAR-10 data (auto-created)
├─ notebooks/
│   └─ 01_explore_data.ipynb # Optional EDA playground
├─ src/
│   ├─ config.py             # Hyperparameters & device selection
│   ├─ dataset.py            # Dataloaders + transforms
│   ├─ model.py              # Simple CNN
│   ├─ train.py              # Train/val loop + checkpointing
│   ├─ evaluate.py           # Load best model and visualize predictions
│   └─ utils.py              # Small helpers (seeding, etc.)
└─ saved_models/
    └─ best_model.pth        # Saved weights (created after training)
```

## Setup
1) Create and activate a virtual environment (recommended).
2) Install dependencies:
```bash
pip install -r requirements.txt
```

## Train
```bash
python -m src.train
```
- Downloads CIFAR-10 to `data/` if needed.
- Trains for `NUM_EPOCHS` (see `src/config.py`), tracking train/val metrics.
- Saves the best checkpoint to `saved_models/best_model.pth`.

## Evaluate / visualize
```bash
python -m src.evaluate
```
Loads the saved checkpoint, runs a quick prediction batch, and shows 8 images with true vs predicted labels.

## Config tweaks
Edit `src/config.py` to adjust device selection, batch size, learning rate, and epoch count. Default prefers Apple `mps` if available.

## Reproducibility
Training seeds Python, NumPy, and PyTorch (`src/utils.py`). GPU/accelerator determinism can vary by hardware and backend.

## Notebook
`notebooks/01_explore_data.ipynb` is a blank shell you can use for quick EDA or sanity checks on the dataset.
