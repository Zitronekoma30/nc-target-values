from typing import Dict, Tuple
import torch
import numpy as np
import torch.nn.functional as F

CLASSES = 10
CLASS_TUPLES = []

for i in range(CLASSES):
    t = [0 for _ in range(CLASSES)]
    t[i] = 1
    CLASS_TUPLES.append(tuple(t))
def one_hot_nll_loss(output, oh_target):
    """Calculates loss using target vectors instead of target ints, returns loss"""
    log_probs = output # NN does log softmax already
    loss = -(oh_target * log_probs).sum(dim=-1).mean()

    return loss

def generate_target(target):
    """Generates one hot encoded target vector from integer target. Returns new OH vector."""
    one_hot = F.one_hot(target, num_classes=CLASSES)
    new_target = one_hot.float()

    return new_target

def generate_alt_target(target, targets):
    out = generate_target(target)
    if targets == "soft":
        out[out == 1.0] = 0.8
        out[out == 0.0] = 0.2
    return out

def apply_class_values(oh_targets: torch.Tensor, nc: Dict[Tuple, float], c: Dict[Tuple, float]) -> torch.Tensor:
    """Switches a batch of target vectors from standard one hot encoding to soft targets determined by provided c and nc."""
    out_targets = []
    # print(f"Input tensor: {oh_targets}")
    for oh_target in oh_targets:
        target = tuple(oh_target.tolist())
        tvec = np.array(target)
        tvec[tvec == 1] = c[target]
        tvec[tvec == 0] = nc[target]
        out_targets.append(tvec)
    # print(f"Output Tensor: {torch.tensor(np.array(out_targets))}")
    return torch.tensor(np.array(out_targets), dtype=torch.float32)

def add_outputs_by_class(outputs, output, target):
    """Groups outputs by target class in a dictionary. Returns the updated dictionary."""
    for single_output, single_target in zip(output, target):
        _class = tuple(F.one_hot(single_target, num_classes=CLASSES).tolist())
        if _class not in outputs:
            outputs[_class] = []
        outputs[_class].append(single_output)
    return outputs

def predict_by_nearest_target(output, class_targets, nc_by_class):
    device, C = output.device, output.size(1)
    template = torch.empty((C, C), dtype=torch.float32, device=device)
    
    for one_hot, log_pc in class_targets.items():
        k = one_hot.index(1)
        row_nc = float(nc_by_class[one_hot])  # Convert to clean float
        template[k].fill_(row_nc)
        template[k, k] = float(log_pc)        # Convert to clean float
    
    diffs = ((output.unsqueeze(1) - template).pow(2).mean(dim=2))
    return diffs.argmin(dim=1)
