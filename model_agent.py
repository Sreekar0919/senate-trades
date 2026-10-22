"""
ModelAgent — trains and executes a linear layer to optimize agent weights.

This agent takes the sub-scores from the 5 main agents as inputs and predicts 
the historical 'excess_return' (alpha) from the trade CSV.
"""
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

log = logging.getLogger("ModelAgent")

class ScoringModel(nn.Module):
    def __init__(self, input_dim=6):
        super(ScoringModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        # Stronger initialization to ensure non-zero starting scores
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[0.10, 0.25, 0.20, 0.10, 0.20, 0.15]]))
            self.linear.bias.fill_(0.1)

    def forward(self, x):
        # Use ReLU to ensure scores stay non-negative
        return torch.relu(self.linear(x))

class ModelAgent:
    NAME = "ModelAgent"
    MODEL_PATH = Path(__file__).parent / "data" / "optimized_weights.pth"

    def __init__(self, input_dim: int = 6):
        self.model = ScoringModel(input_dim=input_dim)
        self._load_model()

    def _load_model(self):
        if self.MODEL_PATH.exists():
            try:
                self.model.load_state_dict(torch.load(self.MODEL_PATH))
                log.info("Loaded optimized weights from disk")
            except:
                log.info("Failed to load model, using defaults")

    def save_model(self):
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.MODEL_PATH)
        log.info("Saved optimized weights to %s", self.MODEL_PATH)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 200):
        """
        Trains the linear layer on historical data.
        X: (N, 5) array of sub-scores.
        y: (N, 1) array of excess returns.
        """
        log.info("Optimizing weights using a Linear Layer (PyTorch)…")
        X_tensor = torch.tensor(X, dtype=torch.float32)
        # Normalize target alpha values to be in a reasonable range (-1.0 to 1.0) for training
        # Most alphas are percentages or raw differences. We'll clip extreme outliers.
        y_scaled = np.clip(y, -100, 100) / 100.0
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            if torch.isnan(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

        # Normalize weights to sum to 1.0 for interpretability
        with torch.no_grad():
            weights = self.model.linear.weight.data
            # Ensure weights are positive
            weights.clamp_(min=0.01)
            total = weights.sum()
            if total > 0 and not torch.isnan(total):
                self.model.linear.weight.data = weights / total
            else:
                # Fallback to predetermined weights if nan occurs
                self.model.linear.weight.copy_(torch.tensor([[0.10, 0.30, 0.25, 0.15, 0.20]]))
            
        self.save_model()
        log.info("Optimization complete. New weights: %s", self.get_weights())

    def get_weights(self) -> list[float]:
        with torch.no_grad():
            return self.model.linear.weight.data.numpy().flatten().tolist()

    def predict(self, features: list[float]) -> float:
        """
        Calculates the optimized score for a given set of sub-scores.
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32)
            score = self.model(x).item()
            return max(0.0, min(score, 1.0))
