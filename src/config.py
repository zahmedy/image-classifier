import torch

# Device: Use Apple GPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available else "cpu")

# Data settings 
DATA_DIR = "./data"     
NUM_CLASSES = 10        # Image classes

# Training settings
BATCH_SIZE = 64
NUM_EPOCHE = 10
LEARNING_RATE = 1e-3

# For reproducibility
RANDOM_SEED = 42


