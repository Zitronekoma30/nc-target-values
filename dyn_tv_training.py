# Initial Code: https://nextjournal.com/gkoehler/pytorch-mnist
import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Dict, Tuple
import wandb
from typing import cast
from collections.abc import Sized
from enum import Enum
import argparse
import adaptations

# ---------------------------------------------- #
#               SETUP / LOADING DATA             #
# ---------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size-train', type=int, default=64)
    parser.add_argument('--batch-size-test', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--adaptation', choices=adaptations.ADAPTATION_REGISTRY.keys(), default="base")
    return parser.parse_args()

args = parse_args()

# set values
n_epochs = args.epochs
batch_size_train = args.batch_size_train
batch_size_test = args.batch_size_test
learning_rate = args.learning_rate
momentum = args.momentum
log_interval = args.log_interval
random_seed = args.seed
adaptation: Callable = adaptations.ADAPTATION_REGISTRY[args.adaptation]

CLASSES = 10

torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# load data
train_loader: torch.utils.data.DataLoader[torchvision.datasets.MNIST] = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,),(0.3081,))
        ])),
        batch_size=batch_size_train, shuffle=True)

test_loader: torch.utils.data.DataLoader[torchvision.datasets.MNIST] = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,),(0.3081,))
        ])),
        batch_size=batch_size_train, shuffle=True)


def show_examples():
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title(f"Ground Truth: {example_targets[i]}")
        plt.xticks([])
        plt.yticks([])
    fig.show()
    input()

# show_examples()

# WEIGHTS AND BIASES EXPERIMENT SETUP
run = wandb.init(
    entity="leon-andrassik-paris-lodron-universit-t-salzburg",
    project="nc-adaptive-tv",

    config=vars(args)
)

##################################
##          Network Setup       ##
##################################

# build NN
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50,10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

# training
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

# ----------------------------------------------------------- #
#               Training / Target Value generation            #
# ----------------------------------------------------------- #

## logging
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader)*batch_size_train for i in range(n_epochs+1)]
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

def apply_class_values(oh_targets: torch.Tensor, nc: float, c: Dict[Tuple, float]) -> torch.Tensor:
    """Switches a batch of target vectors from standard one hot encoding to soft targets determined by provided c and nc."""
    out_targets = []
    # print(f"Input tensor: {oh_targets}")
    for oh_target in oh_targets:
        target = tuple(oh_target.tolist())
        tvec = np.array(target)
        tvec[tvec == 1] = c[target]
        tvec[tvec == 0] = nc
        out_targets.append(tvec)
    # print(f"Output Tensor: {torch.tensor(np.array(out_targets))}")
    return torch.tensor(oh_targets)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA IS AVAILABLE: {torch.cuda.is_available()}")

network = network.to(device)

def add_outputs_by_class(outputs, output, target):
    """Groups outputs by target class in a dictionary. Returns the updated dictionary."""
    for single_output, single_target in zip(output, target):
        _class = tuple(F.one_hot(single_target, num_classes=CLASSES).tolist())
        if _class not in outputs:
            outputs[_class] = []
        outputs[_class].append(single_output)
    return outputs

def pre_train() -> Tuple[float, Dict[Tuple, float]]:
    """Find the naturally preferred values for each class as well as the non class value, returns their mean"""
    network.eval()
    outputs = {}
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = network(data)
            outputs = add_outputs_by_class(outputs, output, target)
        # calc and return class and non class values
        # # Get list of (non) class values
        nc_vals = []
        c_vals = {}

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
        # # average
        nc_mean = np.mean(nc_vals)
        c_means = {}
        for c in c_vals:
            c_means[c] = np.mean(c_vals[c])
            # print(c_vals[c])
        return float(nc_mean), c_means


##############################################
##          TRAINING AND TESTING            ##
##############################################

def train(epoch: int, nc: float, c: Dict[Tuple, float], spacing: Callable = adaptations.base):
    """Trains network according using specified target spacing method, returns new spaced values for next epoch"""
    network.train()
    
    outputs = {}
    class_targets: Dict[Tuple, float] = c
    nclass_target: float = nc

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        # new_target = generate_target(target)
        new_target = apply_class_values(generate_target(target), nclass_target, class_targets).to(device)
        optimizer.zero_grad()
        output = network(data)
        outputs = add_outputs_by_class(outputs, output, target)
        
        loss = one_hot_nll_loss(output, new_target)
        run.log({"loss":loss, "epoch":epoch + batch_idx / len(train_loader)})
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader)*batch_size_train} Loss: {loss.item()}]")

            torch.save(network.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')

    return spacing(class_targets, nclass_target, outputs)



def test(epoch=0.0):
    """Tests network using test dataset"""
    network.eval()
    test_loss = 0
    correct = 0
    total_confidence = 0.0
    total_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target_vecs = generate_target(target).to(device)
            output = network(data)

            loss = one_hot_nll_loss(output, target_vecs)
            test_loss += loss * data.size(0)

            # Prediction and accuracy (same as before)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.to(device).data.view_as(pred)).sum()

            # Confidence: cosine similarity between output and target vector
            cos_sim = torch.nn.functional.cosine_similarity(output, target_vecs, dim=1)
            total_confidence += cos_sim.sum().item()
            total_samples += data.size(0)

    test_loss /= len(cast(Sized, test_loader.dataset))
    acc = 100. * correct / len(cast(Sized, test_loader.dataset))
    avg_conf = total_confidence / total_samples

    run.log({
        "test accuracy (%)": acc,
        "test loss": test_loss,
        "avg cosine similarity (confidence)": avg_conf,
        "epoch": epoch
    })





###############################
##        RUN EXPERIMENT     ##
###############################

nc, c = pre_train()
test(epoch=1)
for epoch in range(1, n_epochs + 1):
    nc, c = train(epoch, nc=nc, c=c, spacing=adaptation)
    test(epoch=float(epoch)+0.999)

run.finish()

