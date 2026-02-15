import os
import torch
import json
from tqdm import tqdm
from dataset import ARCDataset
from architecture import TinyRecursiveARC

def load_model(model_path, device):
    """Loads the TinyRecursiveARC model from a checkpoint."""
    print(f"Loading model from {model_path}...")
    model = TinyRecursiveARC().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded successfully.")
    else:
        print(f"Warning: Checkpoint not found at {model_path}. Using initialized weights.")
    model.eval()
    return model

def evaluate_accuracy(model, dataset, device):
    """Evaluates exact grid match accuracy on the dataset."""
    print(f"Evaluating on {len(dataset)} examples...")
    correct_count = 0
    total_count = len(dataset)

    # Disable augmentation for evaluation
    dataset.augment = False

    with torch.no_grad():
        for i in tqdm(range(total_count), desc="Evaluating"):
            item = dataset[i]

            # Pad input to 30x30 to match training and model expectation
            inp = item['input']
            H, W = inp.shape
            padded_inp = torch.full((30, 30), 10, dtype=torch.long)
            padded_inp[:H, :W] = inp

            padded_inp = padded_inp.unsqueeze(0).to(device) # Add batch dim
            target = item['output'].to(device)

            # Get hard prediction (grid)
            # hard_predict returns a list of tensors (one per batch item)
            preds = model.hard_predict(padded_inp)
            pred_grid = preds[0] # Batch size is 1

            # Check for exact match
            if pred_grid.shape == target.shape:
                if torch.equal(pred_grid, target):
                    correct_count += 1

    accuracy = (correct_count / total_count) * 100.0
    return accuracy

def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "checkpoints/tiny_recursive_arc_v3.pth"

    # Auto-detect data path (Kaggle vs Local)
    kaggle_input = "/kaggle/input/arc-prize-2024"
    if os.path.exists(kaggle_input):
        base_path = kaggle_input
    elif os.path.exists("arc-agi_evaluation_challenges.json"):
        base_path = "."
    else:
        print("Error: Dataset not found.")
        return

    eval_challenges_path = os.path.join(base_path, "arc-agi_evaluation_challenges.json")
    eval_solutions_path = os.path.join(base_path, "arc-agi_evaluation_solutions.json")

    if not os.path.exists(eval_challenges_path) or not os.path.exists(eval_solutions_path):
        print(f"Error: Evaluation files not found at {base_path}")
        return

    # Load Dataset
    # We use the evaluation set
    dataset = ARCDataset(
        challenges_path=eval_challenges_path,
        solutions_path=eval_solutions_path,
        augment=False
    )

    # Load Model
    model = load_model(checkpoint_path, device)

    # Run Evaluation
    accuracy = evaluate_accuracy(model, dataset, device)

    print(f"\nResults:")
    print(f"Total Examples: {len(dataset)}")
    print(f"Exact Match Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
