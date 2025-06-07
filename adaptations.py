from typing import Dict, Tuple, List
import torch
import numpy as np


# ------- Helpers ------- #
def extract_vals(outputs: Dict[Tuple, Tuple]) -> Tuple[Dict[Tuple, List], Dict[Tuple, List]]:
    """Extracts all class and non-class values grouped by class from outputs dict in log p. Returns values in probability space."""
    nc_vals: Dict[Tuple, List] = {}
    c_vals: Dict[Tuple, List] = {}

    for c in outputs:
        cv_idx = c.index(1)
        c_vals[c] = []
        nc_vals[c] = []
        # print(f"Class {c} class values at index {cv_idx}")
        for t in outputs[c]:
            t = t.exp()
            # Class vals
            c_vals[c].append(t[cv_idx].item())
            # Non class
            other_values = torch.cat([t[:cv_idx], t[cv_idx+1:]])

            nc_vals[c] += other_values.tolist()
    return nc_vals, c_vals

#######################################
##          Adaptations              ##
#######################################

def base(class_targets: Dict[Tuple, float], nclass_targets: Dict[Tuple, float], outputs: Dict[Tuple, Tuple], multiplier: float = 1) -> Tuple[Dict[Tuple, float], Dict[Tuple, float]]:
    """Serves as stand in if no spacing is desired, returns back existing nc and c"""
    return  nclass_targets, class_targets

def sigma(class_targets: Dict[Tuple, float], nclass_targets: Dict[Tuple, float], outputs: Dict[Tuple, Tuple], uni_directional = False, multiplier: float = 1) -> Tuple[Dict[Tuple, float], Dict[Tuple, float]]:
    """Spaces the class/non class values using a combination of their std deviation, returns new nc and c"""
    new_nc: Dict[Tuple, float] = nclass_targets.copy() # in prob space
    nc_o, c_o = extract_vals(outputs) # IN p NOT log p

    means_c: Dict[Tuple, float] = {}
    means_nc: Dict[Tuple, float] = {}

    std_devs_c: Dict[Tuple, float] = {}
    std_devs_nc: Dict[Tuple, float] = {}

    for c in c_o.keys():
        std_devs_c[c] = float(np.std(c_o[c]))
        std_devs_nc[c] = float(np.std(nc_o[c]))

        means_c[c] = float(np.mean(c_o[c]))
        means_nc[c] = float(np.mean(nc_o[c]))

    for c in std_devs_c:
        s_sum = std_devs_c[c] + std_devs_nc[c]
        if np.abs(class_targets[c] - new_nc[c]) >= s_sum*multiplier:
            continue
        else:
            class_val = float(class_targets[c])
            non_class_val = float(nclass_targets[c])

            if uni_directional or non_class_val <= class_val:
                print("pushing non-class down")
                adjusted = float(non_class_val - s_sum) * multiplier
                new_nc[c] = max(0.0, adjusted)

                if new_nc[c] < 0.0001:
                    class_targets[c] = 0.0 + s_sum * multiplier
            else:
                print("pushing non-class up")
                adjusted = float(non_class_val + s_sum) * multiplier
                new_nc[c] = min(1.0, adjusted)

                if new_nc[c] > 0.999:
                    class_targets[c] = 1.0 - s_sum * multiplier

    # REMEMBER .exp() for outputs to get to probability space for std dev because of log_softmax on forward
    # .log() to go back to logs
    # use probs for calculation use logs for training
    return new_nc, class_targets

# ---------------- Registry ------------------- #

ADAPTATION_REGISTRY = {
    "base": base,
    "sigma": sigma,
    # Add more strategies here
}
