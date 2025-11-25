import torch
import matplotlib.pyplot as plt

from .dataset import get_dataloaders
from .model import SimpleCNN
from .config import DEVICE

# CIFAR-10 class names
CLASSES = [
    'airplane', 'automobile', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]


def show_batch(images, labels, preds=None, max_images=8):
    """
    Visualize a small batch of images.

    If preds is given, it will show: True (Pred) in the title.
    Otherwise, it just shows the true label.
    """

    images = images.cpu()
    labels = labels.cpu()
    if preds is not None:
        preds = preds.cpu()

    # CIFAR-10 normalization we applied earlier:
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    num_images = min(max_images, images.size(0))
    plt.figure(figsize=(12, 4))

    for i in range(num_images):
        img = images[i] * std + mean # de-normalize
        img = img.permute(1, 2, 0)  # CHW -> HWC

        label_idx = labels[i].item()
        true_label = CLASSES[label_idx]

        if preds is not None:
            pred_idx = preds[i].item()
            pred_label = CLASSES[pred_idx]
            title = f"T: {true_label}\nP: {pred_label}"
        else:
            title = f"T: {true_label}"

        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def main():
    # Get data loaders (only need test_loader here)
    _, _, test_loader = get_dataloaders()

    # Load model and best weights
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load("saved_models/best_model.pth", map_location=DEVICE))
    model.eval()

    # Take one batch from test set
    images, labels = next(iter(test_loader))
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    with torch.no_grad():
        output = model(images)
        _, preds = torch.max(output, dim=1)

    # Show images with true vs predicted labels 
    show_batch(images, labels, preds, max_images=8)


if __name__ == "__main__":
    main()