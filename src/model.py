import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_CLASSES, DEVICE


class SimpleCNN(nn.Module):
    """
    A Simple Convolutional Neural Network for CIFAR-10

    Input: (B, 3, 32, 32)
    Output: (B, Num_class) =)(B, 10) logits
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # ------- Convolutional feature extractor -------
        # Each block:
        #   Conv2d -> ReLU -> MaxPool2d
        self.conv_layers = nn.Sequential(
                # Block 1: 3 x 32 x 32 -> 32 x 32 x 32 -> 32 x 16 x 16
                nn.Conv2d(
                    in_channels=3,
                    out_channels=32, 
                    kernel_size=3,
                    padding=1
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # Block 2: 32 x 16 x 16 -> 64 x 16 x 16 -> 64 x 8 x 8
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=3,
                    padding=1
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), # 16 -> 8

                # Block 3: 64 x 8 x 8 -> 128 x 8 x 8 -> 128 x 4 x 4
                nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=2,
                    padding=1
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)   # 8 -> 4
            )
        
        # After 3 blocks:
        # feature map shape = (128, 4, 4)
        # flattened size = 128 * 4 * 4 = 2048
        self.flatten_dim = 128 * 4 * 4

        # ------- Fully-connected classifier -------
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        """
        x: input images, shape (B, 3, 32, 32)
        returns: logits, shape (B, NUM_CLASSES)
        """
        x = self.conv_layers(x)  # (B, 128, 4, 4)

        # Flatten: keep batch dimension, collapse the rest
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc_layers(x)

        return x 


if __name__ == "__main__":
    model = SimpleCNN().to(DEVICE)
    dummy_input = torch.randn(4,3,32,32).to(DEVICE) # batch of 4 random images
    output = model(dummy_input)
    print("Output shape:", output.shape) # expected: torch.Size([4, NUM_CLASSES])