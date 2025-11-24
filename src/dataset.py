import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from .config import DATA_DIR, BATCH_SIZE, RANDOM_SEED

# Define transformer for CIFAR-10
#       - ToTensor: converts PIL image (0-255) -> tensor (0-1)
#       - Normalize: shifts and scales each channel (R,G,B) to have mean ~0, std ~1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])


def get_dataloaders(val_ratio: float = 0.1):
    """
    Download CIFAR-10 (if needed), apply transforms, and return 
    train, val, and test Dataloader.

    Returns:
        train_loader, val_loader, test_loader
    """

    full_train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    # split to train and validation
    torch.manual_seed(RANDOM_SEED)
    num_train = len(full_train_dataset)
    num_val = int(num_train * val_ratio)
    num_train = num_train - num_val

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [num_train, num_val]
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    # wrap with DataLoader: handles batching and shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print("Batch images shape:", images.shape)   # expected: [BATCH_SIZE, 3, 32, 32]
    print("Batch labels shape:", labels.shape)   # expected: [BATCH_SIZE]