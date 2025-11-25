import os
import torch
import torch.nn as nn
import torch.optim as optim

from .dataset import get_dataloaders
from .model import SimpleCNN
from .config import DEVICE, NUM_EPOCHS, LEARNING_RATE, RANDOM_SEED
from .utils import set_seed



def train_one_epoch(model, dataloader, criterion, optimizer):
    """
    Train the model for ONE epoch over the given dataloader

    Returns:
        avg_loss, avg_accuracy
    """
    model.train() # set the model in training mode

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in dataloader:
        # Move data to correct device
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward pass: get model predictions for this batch
        outputs = model(images)  # shape: (B, NUM_CLASSES)
        
        # Compute loss
        loss = criterion(outputs, labels)

        # zero the gradient
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # update weights
        optimizer.step()

        # --------- Stats for monitoring ---------
        running_loss += loss.item() * images.size(0) # sum of loss over batch

        # Predicted class is the index with highest logit
        _, preds = torch.max(outputs, dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    avg_acc = running_correct / total

    return avg_loss, avg_acc


def evaluate(model, dataloader, criterion):
    """
    Evaluate the model (no gradient updates).

    Returns:
        avg_loss, avg_accuracy
    """
    model.eval()    # important: eval mode (no dropout, no batchnorm update)

    running_loss = 0.0
    running_correct = 0
    total = 0

    # We don't need gradients during evaluation
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, dim=1)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    avg_acc = running_correct / total

    return avg_loss, avg_acc


def main():
    set_seed(RANDOM_SEED)
    # Get dataloader
    train_loader, val_loader, test_loader = get_dataloaders()

    # create model and move to device
    model = SimpleCNN().to(DEVICE)
    print(f"Using device: {DEVICE}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()           # Standard for classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs("saved_models", exist_ok=True)
    best_val_acc = 0.0

    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion
        )

        print(f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} Val Acc:   {val_acc:.4f}")

        # Save best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "saved_models/best_model.pth")
            print(f"✅ New best model saved with val acc = {best_val_acc:.4f}")

    # Final evaluation on test set
    print("\nEvaluating on test set with best model....")
    best_model = SimpleCNN().to(DEVICE)
    best_model.load_state_dict(torch.load("saved_models/best_model.pth", map_location=DEVICE))

    test_loss, test_acc = evaluate(best_model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
