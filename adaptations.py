from typing import Dict, Tuple, List
import torch
import numpy as np
from enum import Enum


# ------- Helpers ------- #
def extract_vals(outputs: Dict[Tuple, Tuple]):
    """Extracts all class and non-class values grouped by class from outputs dict."""
    nc_vals: List[float] = []
    c_vals: Dict[Tuple, List] = {}

    for c in outputs:
        cv_idx = c.index(1)
        c_vals[c] = []
        # print(f"Class {c} class values at index {cv_idx}")
        for t in outputs[c]:
            t = t.exp()
            # Class vals
            c_vals[c].append(t[cv_idx].item())
            # Non class
            other_values = torch.cat([t[:cv_idx], t[cv_idx+1:]])
            nc_vals += other_values.tolist()
    return nc_vals, c_vals

#######################################
##          Adaptations              ##
#######################################

def base(class_targets: Dict[Tuple, float], nclass_target: float, outputs: Dict[Tuple, Tuple]):
    """Serves as stand in if no spacing is desired, returns back existing nc and c"""
    return  nclass_target, class_targets

def sigma(class_targets: Dict[Tuple, float], nclass_target: float, outputs: Dict[Tuple, Tuple]):
    """Spaces the class/non class values using a combination of their std deviation, returns new nc and c"""
    std_devs_class = {}
    std_dev_nc = 0
    # REMEMBER .exp() for outputs to get to probability space for std dev because of log_softmax on forward
    # .log() to go back to logs
    # use probs for calculation use logs for training
    new_class, new_nclass = 0, 0
    return new_nclass, new_class

# ---------------- Registry ------------------- #

class AdaptationStrategy(Enum):
    BASE = "base"
    SIGMA = "sigma"  # Add more as needed
    # Example: ADVANCED = "advanced"

ADAPTATION_REGISTRY = {
    "base": base,  # Replace with actual function
    "sigma": sigma,
    # Add more strategies here
}
