import numpy as np
import adaptations
from typing import Tuple, Dict, Callable
import utils
import torch

def uniform(config, network=None, test_data=None, device=None, spacing=adaptations.base):
    c_vals = {}
    nc_vals = {}
    
    for class_tuple in utils.CLASS_TUPLES:
        c_vals[class_tuple] = np.random.uniform(0, 1)
        nc_vals[class_tuple] = np.random.uniform(0, 1)
    
    return spacing(c_vals, nc_vals, {}, multiplier=config.first_push_multiplier)

def average(config, network, test_data, device, spacing=adaptations.base):
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

def soft(config, network=None, test_data=None, device=None, spacing=adaptations.base):
    c_vals = {}
    nc_vals = {}
    
    for class_tuple in utils.CLASS_TUPLES:
        c_vals[class_tuple] = 0.8
        nc_vals[class_tuple] = 0.2
    
    return spacing(c_vals, nc_vals, {}, multiplier=config.first_push_multiplier)

PRETRAINING_REGISTRY = {
    "uniform": uniform,
    "average": average,
    "soft": soft,
    # Add more strategies here
}
