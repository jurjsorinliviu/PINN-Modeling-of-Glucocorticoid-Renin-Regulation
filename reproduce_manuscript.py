import torch
import numpy as np
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

from src.data import prepare_training_data
from src.model import ReninPINN
from src.trainer import PINNTrainer

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ONLY original experimental data (no synthetic)
data = prepare_training_data(dataset='elisa', use_log_scale=False)

# Initialize model (same architecture as manuscript)
model = ReninPINN(
    hidden_layers=[128, 128, 128, 128],
    activation='tanh'
)

# Initialize standard trainer (NO enhanced constraints)
trainer = PINNTrainer(
    model=model,
    device=device,
    learning_rate=1e-3,
    weight_decay=0.01
)

# Train with standard settings (NO two-stage, NO synthetic data)
trainer.train(
    data_dict=data,
    n_epochs=10000,
    print_every=1000,
    curriculum_learning=True
)

# Evaluate
results = trainer.evaluate(data)
print(f"R²: {results['r_squared']:.4f}")
print(f"RMSE: {results['rmse']:.4f}")
print(f"IC50: {results['ic50']:.3f}")
print(f"Hill: {results['hill']:.3f}")

# Save model
trainer.save_checkpoint('manuscript_model.pth')