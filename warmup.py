import numpy as np
import adaptations
from typing import Tuple, Dict, Callable, List
import utils
import torch

def uniform(config, network=None, test_data=None, device=None, spacing=adaptations.base, nudge=2.0):
    c_vals = {}
    nc_vals = {}
    
    for class_tuple in utils.CLASS_TUPLES:
        c_vals[class_tuple] = np.random.uniform(0, 1)
        nc_vals[class_tuple] = np.random.uniform(0, 1)
    
    return spacing(c_vals, nc_vals, {}, multiplier=config.first_push_multiplier)

def average(config, network, test_data, device, spacing=adaptations.base, nudge=2.0) -> Tuple[Dict[Tuple, float], Dict[Tuple, float]]:
    """Find the naturally preferred values for each class as well as the non class value, returns their mean in p space"""
    network.eval()
    outputs = {}
    with torch.no_grad():
        for data, target in test_data:
            data = data.to(device)
            output = network(data)
            outputs = utils.add_outputs_by_class(outputs, output, target)
        # calc and return class and non class values
        # # Get list of (non) class values
        nc_vals, c_vals = adaptations.extract_vals(outputs)

        nc_means = {}
        c_means = {}
        for c in c_vals:
            c_means[c] = np.mean(c_vals[c])
            nc_means[c] = np.mean(nc_vals[c])
            # print(c_vals[c])

        # return np.log(1), c_means
        return spacing(c_means, nc_means, outputs, multiplier=config.first_push_multiplier)

def average_nudge(config, network, test_data, device, spacing=adaptations.base, nudge=0.2):
    non_class_values, class_values = average(config, network, test_data, device, spacing)
    for c in class_values:
        if non_class_values[c] > class_values[c]:
            non_class_values[c] += nudge
        else:
            non_class_values[c] -= nudge

        if non_class_values[c] <= 0:
            non_class_values[c] = 0
            class_values[c] = min(nudge, 1)
        elif non_class_values[c] >= 1:
            non_class_values[c] = 1
            class_values[c] = max(1 - nudge, 0)

    return non_class_values, class_values


def average_nudge_remap(config, network, test_data, device, spacing=adaptations.base, nudge=0.2):
    non_class_values, class_values = average_nudge(config, network, test_data, device, spacing, nudge)

    all_vals: List[float] = list(non_class_values.values()) + list(class_values.values())
    all_vals.sort()

    scaling_factor = 1/all_vals[-1]

    for c in class_values:
        class_values[c] = class_values[c] * scaling_factor
        non_class_values[c] = non_class_values[c] * scaling_factor

    return non_class_values, class_values

def soft(config, network=None, test_data=None, device=None, spacing=adaptations.base, nudge=2.0):
    c_vals = {}
    nc_vals = {}
    
    for class_tuple in utils.CLASS_TUPLES:
        c_vals[class_tuple] = 0.8
        nc_vals[class_tuple] = 0.2
    
    return spacing(c_vals, nc_vals, {}, multiplier=config.first_push_multiplier)

PRETRAINING_REGISTRY = {
    "uniform": uniform,
    "average": average,
    "average_nudge": average_nudge,
    "average_nudge_remap": average_nudge_remap,
    "soft": soft,
    # Add more strategies here
}
