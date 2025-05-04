from typing import Dict, Tuple
import torch
import numpy as np
from enum import Enum


#######################################
##          Adaptations              ##
#######################################

def base(class_targets: Dict[Tuple, float], nclass_target: float, outputs):
    """Serves as stand in if no spacing is desired, returns back existing nc and c"""
    return  nclass_target, class_targets

def sigma(class_targets: Dict[Tuple, float], nclass_target: float, outputs):
    """Spaces the class/non class values using a combination of their std deviation, returns new nc and c"""
    std_devs_class = {}
    std_dev_nc = 0
    # REMEMBER .exp() to get to probability space for std dev
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
