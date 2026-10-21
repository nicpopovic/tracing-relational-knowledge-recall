"""
Linear probe implementation for relation classification.
"""

import torch
import torch.nn as nn
from typing import Optional


class LinearProbe(nn.Module):
    """
    Simple linear probe for relation classification.

    Maps from feature space to relation classes using a linear transformation.
    """

    def __init__(self, input_dim: int, num_classes: int):
        """
        Initialize the linear probe.

        Args:
            input_dim: Dimensionality of input features
            num_classes: Number of relation classes to predict
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape [batch_size, input_dim]

        Returns:
            Logits of shape [batch_size, num_classes]
        """
        return self.linear(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (argmax of logits).

        Args:
            x: Input features

        Returns:
            Predicted class indices
        """
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=-1)


def train_probe(
    probe: LinearProbe,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    num_epochs: int = 200,
    lr: float = 1e-2,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> LinearProbe:
    """
    Train a linear probe using Adam optimizer and cross-entropy loss.

    Args:
        probe: LinearProbe instance to train
        X_train: Training features [N, D]
        y_train: Training labels [N]
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        verbose: Whether to print training progress

    Returns:
        Trained probe
    """
    if device is None:
        device = X_train.device

    probe.to(device)
    probe.train()

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = probe(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 50 == 0:
            acc = (probe.predict(X_train) == y_train).float().mean().item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")

    return probe


def evaluate_probe(
    probe: LinearProbe,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> dict:
    """
    Evaluate a trained probe.

    Args:
        probe: Trained LinearProbe
        X_test: Test features [N, D]
        y_test: Test labels [N]

    Returns:
        Dictionary with evaluation metrics
    """
    probe.eval()

    with torch.no_grad():
        predictions = probe.predict(X_test)
        correct = (predictions == y_test).sum().item()
        total = y_test.numel()
        accuracy = correct / total

        logits = probe(X_test)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y_test).item()

    return {
        "accuracy": accuracy,
        "loss": loss,
        "correct": correct,
        "total": total,
    }
