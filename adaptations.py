from typing import Dict, Tuple, List
import torch
import numpy as np
from enum import Enum


# ------- Helpers ------- #
def extract_vals(outputs: Dict[Tuple, Tuple]) -> Tuple[List[float], Dict[Tuple, List]]:
    """Extracts all class and non-class values grouped by class from outputs dict in log p. Returns values in probability space."""
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

def base(class_targets: Dict[Tuple, float], nclass_target: float, outputs: Dict[Tuple, Tuple]) -> Tuple[float, Dict[Tuple, float]]:
    """Serves as stand in if no spacing is desired, returns back existing nc and c"""
    return  nclass_target, class_targets

def sigma(class_targets: Dict[Tuple, float], nclass_target: float, outputs: Dict[Tuple, Tuple]) -> Tuple[float, Dict[Tuple, float]]:
    """Spaces the class/non class values using a combination of their std deviation, returns new nc and c"""
    nc_o, c_o = extract_vals(outputs) # IN p NOT log p

    std_devs_c: Dict[Tuple, float] = {}
    std_dev_nc = np.std(nc_o)

    for c in c_o.keys():
        std_devs_c[c] = float(np.std(c_o[c]))

    # REMEMBER .exp() for outputs to get to probability space for std dev because of log_softmax on forward
    # .log() to go back to logs
    # use probs for calculation use logs for training
        return nclass_target, class_targets

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
