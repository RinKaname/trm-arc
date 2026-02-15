import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import json

# Import our custom modules
from architecture import TinyRecursiveARC
from dataset import ARCDataset, collate_fn

# ==========================================
# Configuration & Hyperparameters
# ==========================================
CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 100,  # Increase for real training
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 2,
    "save_dir": "checkpoints",
    "model_name": "tiny_recursive_arc_v3.pth",
    "seed": 42,
    "data_path": ".",  # Will be auto-detected
}

# ==========================================
# Path Detection (Kaggle vs Local)
# ==========================================
def get_data_paths():
    """Auto-detects data paths for Kaggle or Local environment."""
    kaggle_input = "/kaggle/input/arc-prize-2024"
    local_input = "."

    # Check for Kaggle directory structure
    if os.path.exists(kaggle_input):
        print(f"Detected Kaggle environment. Using input from: {kaggle_input}")
        base_path = kaggle_input
        output_dir = "/kaggle/working"
    elif os.path.exists("arc-agi_training_challenges.json"):
        print("Detected local environment with extracted files.")
        base_path = "."
        output_dir = "checkpoints"
    else:
        print("Warning: Could not find dataset. Please ensure files are in the current directory or /kaggle/input.")
        base_path = "."
        output_dir = "checkpoints"

    # Define specific file paths
    paths = {
        "train_challenges": os.path.join(base_path, "arc-agi_training_challenges.json"),
        "train_solutions": os.path.join(base_path, "arc-agi_training_solutions.json"),
        "eval_challenges": os.path.join(base_path, "arc-agi_evaluation_challenges.json"),
        "eval_solutions": os.path.join(base_path, "arc-agi_evaluation_solutions.json"),
        "output_dir": output_dir
    }
    return paths

# ==========================================
# Training Function
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion_color, criterion_size, device):
    model.train()
    total_loss = 0

    # Use tqdm if interactive, else simple print
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        inputs = batch['input'].to(device)
        targets_color = batch['output'].to(device)
        targets_size = batch['target_size'].to(device).float()

        optimizer.zero_grad()

        # Forward pass with Deep Supervision (returns list of predictions)
        predictions = model(inputs)

        loss = 0
        num_steps = len(predictions)

        # Iterate over all recursion steps and sum the loss
        for step_idx, (logits, pred_size) in enumerate(predictions):
            # Color Loss (Cross Entropy)
            # Flatten: (B, 900, 10) -> (B*900, 10)
            loss_color = criterion_color(logits.view(-1, 10), targets_color.view(-1))

            # Size Loss (MSE)
            loss_size = criterion_size(pred_size, targets_size)

            # Weighted sum (can adjust weights if needed)
            loss += loss_color + loss_size

        # Average loss over steps to keep scale consistent
        loss = loss / num_steps

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(loader)

# ==========================================
# Validation Function
# ==========================================
def validate(model, loader, criterion_color, criterion_size, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch['input'].to(device)
            targets_color = batch['output'].to(device)
            targets_size = batch['target_size'].to(device).float()

            # Forward pass (returns list)
            predictions = model(inputs)

            # Evaluate only on the FINAL step for validation metrics
            final_logits, final_size = predictions[-1]

            loss_color = criterion_color(final_logits.view(-1, 10), targets_color.view(-1))
            loss_size = criterion_size(final_size, targets_size)

            loss = loss_color + loss_size
            total_loss += loss.item()

    return total_loss / len(loader)

# ==========================================
# Main Execution
# ==========================================
def main():
    # 1. Setup
    paths = get_data_paths()
    os.makedirs(paths['output_dir'], exist_ok=True)
    device = torch.device(CONFIG['device'])
    torch.manual_seed(CONFIG['seed'])

    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    if not os.path.exists(paths['train_challenges']):
        print(f"Error: Training file not found at {paths['train_challenges']}")
        return

    full_dataset = ARCDataset(
        challenges_path=paths['train_challenges'],
        solutions_path=paths['train_solutions'],
        augment=True
    )

    # Split into Train/Val (90/10 split)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=CONFIG['num_workers']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=CONFIG['num_workers']
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 3. Model Initialization
    model = TinyRecursiveARC().to(device)
    print("Model initialized.")

    # 4. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    criterion_color = nn.CrossEntropyLoss(ignore_index=10) # 10 is padding
    criterion_size = nn.MSELoss()

    # 5. Training Loop
    best_val_loss = float('inf')
    save_path = os.path.join(paths['output_dir'], CONFIG['model_name'])

    print(f"Starting training for {CONFIG['epochs']} epochs...")
    for epoch in range(CONFIG['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_color, criterion_size, device)
        val_loss = validate(model, val_loader, criterion_color, criterion_size, device)

        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Model saved to {save_path}")

    print("Training complete.")

if __name__ == "__main__":
    main()
