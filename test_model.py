import torch
import torch.nn as nn
from architecture import TinyRecursiveARC

def test_model_forward():
    print("Testing TinyRecursiveARC Forward Pass with Deep Supervision...")

    # Initialize model with default parameters
    model = TinyRecursiveARC()
    print("Model initialized.")

    # Create dummy input: Batch size 2, 30x30 grid
    # Input values are integers 0-10 (10 is padding/color)
    B = 2
    dummy_input = torch.randint(0, 11, (B, 30, 30))
    print(f"Dummy input shape: {dummy_input.shape}")

    try:
        # Run forward pass (now returns a list of predictions)
        predictions = model(dummy_input)

        print("Forward pass successful.")
        print(f"Number of recursion steps (T): {len(predictions)}")

        # Verify length matches T
        if len(predictions) != model.T:
             print(f"ERROR: Expected {model.T} steps, got {len(predictions)}")
        else:
             print("SUCCESS: Output steps match configuration.")

        # Check the last prediction
        logits, pred_size = predictions[-1]
        print(f"Final Logits shape: {logits.shape}")
        print(f"Final Predicted size shape: {pred_size.shape}")

        # Check for NaNs
        if torch.isnan(logits).any() or torch.isnan(pred_size).any():
            print("WARNING: NaNs detected in output!")
        else:
            print("No NaNs detected.")

        # Check output shapes
        # Logits should be (B, 900, 10) - color head output
        expected_logits_shape = (B, 900, 10)
        if logits.shape == expected_logits_shape:
            print("Logits shape matches expected.")
        else:
            print(f"Logits shape mismatch! Expected {expected_logits_shape}, got {logits.shape}")

        # Pred size should be (B, 2)
        expected_size_shape = (B, 2)
        if pred_size.shape == expected_size_shape:
            print("Predicted size shape matches expected.")
        else:
            print(f"Predicted size shape mismatch! Expected {expected_size_shape}, got {pred_size.shape}")

        # Test hard_predict (which uses forward internally)
        print("\nTesting hard_predict...")
        preds = model.hard_predict(dummy_input)
        print(f"hard_predict returned {len(preds)} predictions.")
        print(f"Prediction 0 shape: {preds[0].shape}")

    except Exception as e:
        print(f"ERROR during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_forward()
