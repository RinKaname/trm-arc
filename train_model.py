import torch
import torch.nn as nn
import torch.optim as optim
from architecture import TinyRecursiveARC

def train_step(model, inputs, targets_color, targets_size, criterion_color, criterion_size, optimizer):
    model.train()
    optimizer.zero_grad()

    # Forward pass returns a list of predictions for Deep Supervision
    predictions = model(inputs)

    total_loss = 0
    num_steps = len(predictions)

    # Deep Supervision: Sum loss over all steps
    for step_logits, step_size in predictions:
        # Flatten for loss calculation
        # step_logits: (B, 900, 10) -> (B*900, 10)
        # targets_color: (B, 30, 30) -> (B*900)
        loss_color = criterion_color(step_logits.view(-1, 10), targets_color.view(-1))

        # step_size: (B, 2)
        # targets_size: (B, 2)
        loss_size = criterion_size(step_size, targets_size.float())

        total_loss += loss_color + loss_size

    # Average loss over steps (optional, but keeps scale consistent)
    final_loss = total_loss / num_steps

    final_loss.backward()
    optimizer.step()

    return final_loss.item()

def main():
    print("Initializing TinyRecursiveARC with Deep Supervision...")
    model = TinyRecursiveARC()

    # Dummy data for demonstration
    B = 4
    inputs = torch.randint(0, 11, (B, 30, 30))
    targets_color = torch.randint(0, 10, (B, 30, 30))
    targets_size = torch.tensor([[30, 30] for _ in range(B)])

    criterion_color = nn.CrossEntropyLoss()
    criterion_size = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print("Starting training step...")
    loss = train_step(model, inputs, targets_color, targets_size, criterion_color, criterion_size, optimizer)
    print(f"Training step completed. Loss: {loss:.4f}")

    # Verify Deep Supervision Logic
    # We expect model.T predictions
    predictions = model(inputs)
    print(f"Number of recursion steps (T): {model.T}")
    print(f"Number of predictions returned: {len(predictions)}")

    if len(predictions) == model.T:
        print("SUCCESS: Deep Supervision outputs match recursion depth.")
    else:
        print("FAILURE: Mismatch in prediction steps.")

if __name__ == "__main__":
    main()
