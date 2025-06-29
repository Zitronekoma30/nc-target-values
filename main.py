import argparse
import torch
import adaptations
import models
from data import load_data
from experiments import run_experiment
import warmup

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size-train', type=int, default=64)
    parser.add_argument('--batch-size-test', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--nudge', type=float, default=0.2)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--adaptation', choices=adaptations.ADAPTATION_REGISTRY.keys(), default="base")
    parser.add_argument('--targets', choices=["onehot", "soft", "dynamic"], default="dynamic")
    parser.add_argument('--push-multiplier', type=float, default=1.0)
    parser.add_argument('--first-push-multiplier', type=float, default=1.0)
    parser.add_argument('--warmup', choices=warmup.PRETRAINING_REGISTRY.keys(), default="average")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"pushing by x{args.push_multiplier} and initially by {args.first_push_multiplier} if applicable")

    # Disable randomness for reproducability
    torch.backends.cudnn.enabled = True
    torch.manual_seed(args.seed)

    # build the NN
    model = models.Cnn()

    # Get MNIST images
    train_data, test_data = load_data(args)

    # Run the experiment
    run_experiment(model, train_data, test_data, args)
