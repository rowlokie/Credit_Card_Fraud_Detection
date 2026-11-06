"""
CLI Prediction Script
=====================
Predict whether a single credit card transaction is fraudulent.

Usage:
    # Test with a random sample from the dataset
    python predict.py --sample random

    # Test with a pre-built fraud example
    python predict.py --sample fraud

    # Test with a pre-built legitimate example
    python predict.py --sample legitimate

    # Predict with custom values
    python predict.py --amount 149.62 --time 0

    # Choose a specific model
    python predict.py --sample random --model random_forest
"""

import argparse
import sys
import numpy as np
import pandas as pd

from src.predictor import FraudPredictor, get_sample_transactions
from src.config import RAW_DATA_PATH, FEATURE_COLS


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict if a credit card transaction is fraudulent."
    )
    parser.add_argument(
        "--model", type=str, default="xgboost",
        choices=["xgboost", "random_forest"],
        help="Model to use for prediction (default: xgboost)."
    )
    parser.add_argument(
        "--sample", type=str, default=None,
        choices=["random", "fraud", "legitimate"],
        help="Use a sample transaction ('random', 'fraud', 'legitimate')."
    )
    parser.add_argument("--time", type=float, default=None, help="Transaction time.")
    parser.add_argument("--amount", type=float, default=None, help="Transaction amount.")

    # V1-V28 features
    for i in range(1, 29):
        parser.add_argument(f"--v{i}", type=float, default=None, help=f"PCA feature V{i}.")

    return parser.parse_args()


def get_random_sample():
    """Get a random transaction from the dataset."""
    if not RAW_DATA_PATH.exists():
        print("❌ Dataset not found. Cannot sample. Use --sample fraud/legitimate instead.")
        sys.exit(1)

    df = pd.read_csv(RAW_DATA_PATH)
    sample = df.sample(1, random_state=np.random.randint(0, 10000))
    actual_class = sample["Class"].values[0]

    transaction = {}
    for col in FEATURE_COLS:
        transaction[col] = float(sample[col].values[0])

    return transaction, actual_class


def main():
    """Run prediction."""
    args = parse_args()

    print("\n" + "═" * 60)
    print("  🔍 CREDIT CARD FRAUD PREDICTION")
    print("═" * 60)

    # Load predictor
    try:
        predictor = FraudPredictor(model_name=args.model)
        print(f"\n  Model: {args.model}")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    # Get transaction data
    actual_class = None

    if args.sample:
        if args.sample == "random":
            print("  Source: Random sample from dataset\n")
            transaction, actual_class = get_random_sample()
        else:
            samples = get_sample_transactions()
            transaction = samples[args.sample]
            actual_class = 1 if args.sample == "fraud" else 0
            print(f"  Source: Pre-built {args.sample} example\n")
    else:
        # Build from CLI arguments
        transaction = {}
        if args.time is not None:
            transaction["Time"] = args.time
        else:
            transaction["Time"] = 0.0

        if args.amount is not None:
            transaction["Amount"] = args.amount
        else:
            transaction["Amount"] = 0.0

        for i in range(1, 29):
            val = getattr(args, f"v{i}")
            transaction[f"V{i}"] = val if val is not None else 0.0

        print("  Source: Custom input\n")

    # Print transaction details
    print("  📋 Transaction Details:")
    print(f"    Time:   {transaction['Time']:.2f}")
    print(f"    Amount: ${transaction['Amount']:.2f}")

    # Show non-zero V features
    non_zero_v = {k: v for k, v in transaction.items()
                  if k.startswith("V") and abs(v) > 0.01}
    if non_zero_v:
        top_features = sorted(non_zero_v.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        print(f"    Top features: {', '.join(f'{k}={v:.2f}' for k, v in top_features)}")

    # Make prediction
    print("\n" + "─" * 60)
    result = predictor.predict_single(transaction)

    if result["is_fraud"]:
        print(f"\n  🔴 FRAUD DETECTED")
        print(f"     Confidence: {result['confidence']:.1%}")
        print(f"     Fraud Probability: {result['fraud_probability']:.4f}")
    else:
        print(f"\n  🟢 LEGITIMATE TRANSACTION")
        print(f"     Confidence: {result['confidence']:.1%}")
        print(f"     Fraud Probability: {result['fraud_probability']:.4f}")

    # Show actual class if known
    if actual_class is not None:
        actual_label = "Fraud" if actual_class == 1 else "Legitimate"
        correct = (result["is_fraud"] == (actual_class == 1))
        status = "✅ Correct" if correct else "❌ Incorrect"
        print(f"\n  Actual: {actual_label}  →  {status}")

    print("\n" + "═" * 60 + "\n")


if __name__ == "__main__":
    main()
