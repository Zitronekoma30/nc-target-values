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
    parser.add_argument('--targets', choices=["onehot", "soft", "dynamic"], default="dynamic")
    parser.add_argument('--push-multiplier', type=float, default=1.0)
    parser.add_argument('--first-push-multiplier', type=float, default=1.0)
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
targets = args.targets
push_mult = args.push_multiplier
first_push_mult = args.first_push_multiplier

print(f"pushing by x{push_mult} and initially by {first_push_mult} if applicable")

CLASSES = 10

torch.backends.cudnn.enabled = True
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
conf = vars(args)
conf["nc_type"] = "many"

run = wandb.init(
    entity="leon-andrassik-paris-lodron-universit-t-salzburg",
    project="nc-adaptive-tv",

    config = conf
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
        x = F.sigmoid(F.max_pool2d(self.conv1(x), 2))
        x = F.sigmoid(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.sigmoid(self.fc1(x))
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

def pre_train(spacing: Callable = adaptations.base) -> Tuple[Dict[Tuple, float], Dict[Tuple, float]]:
    """Find the naturally preferred values for each class as well as the non class value, returns their mean in p space"""
    network.eval()
    outputs = {}
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = network(data)
            outputs = add_outputs_by_class(outputs, output, target)
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
        return spacing(c_means, nc_means, outputs, multiplier=first_push_mult)


def predict_by_nearest_target(
    output: torch.Tensor,                        # [B, C]  log-softmax
    class_targets: Dict[Tuple[int, ...], float], # {(1,0,…): log p_hit, …}
    nc_by_class: Dict[Tuple[int, ...], float],   # {(1,0,…): log p_miss, …}
) -> torch.Tensor:
    device, C = output.device, output.size(1)

    # ------------------------------------------------------------------
    # Build template: one row per class, each with its own nc background
    # ------------------------------------------------------------------
    template = torch.empty((C, C), dtype=torch.float32, device=device)

    for one_hot, log_pc in class_targets.items():
        k        = one_hot.index(1)          # class index
        row_nc   = nc_by_class[one_hot]      # that class’s “miss” value
        template[k].fill_(row_nc)            # fill the whole row first
        template[k, k] = log_pc              # then set the diagonal slot

    # ------------------------------------------------------------------
    # Same distance-and-argmin as before
    # ------------------------------------------------------------------
    diffs = ((output.unsqueeze(1) - template)  # [B,1,C] – [1,C,C]
             .pow(2)
             .mean(dim=2))                    # [B,C]
    return diffs.argmin(dim=1)                # [B]


##############################################
##          TRAINING AND TESTING            ##
##############################################

def train(epoch: int, nc: Dict[Tuple, float], c: Dict[Tuple, float], spacing: Callable = adaptations.base, targets="dynamic"):
    """Trains network according using specified target spacing method, returns new spaced values for next epoch"""
    network.train()
    
    outputs = {}
    class_targets: Dict[Tuple, float] = c
    nclass_targets: Dict[Tuple, float] = nc

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        # new_target = generate_target(target)
        if targets != "dynamic":
            new_target = generate_alt_target(target, targets).to(device)
        else:
            new_target = apply_class_values(generate_target(target), nclass_targets, class_targets).to(device)
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

    return spacing(class_targets, nclass_targets, outputs, push_mult)



def test(nc, c, epoch=0.0, targets="dynamic"):
    """Tests network using test dataset"""
    network.eval()

    oh_classes = list(c.keys())
    class_target_vectors = torch.tensor(oh_classes, dtype=torch.float32)
    if targets == "dynamic":
        class_target_vectors = apply_class_values(class_target_vectors, nc, c)
    elif targets == "soft":
        class_target_vectors[class_target_vectors == 1] = 0.8
        class_target_vectors[class_target_vectors == 1] = 0.2

    print(f"class target vectors: {class_target_vectors}")

    for class_vector in c:
        num = class_vector.index(1)
        run.log({
            f"class value ({num})": c[class_vector],
            f"non-class value ({num})": nc[class_vector],
            "epoch": epoch
        })

    test_loss = 0
    total_confidence = 0.0
    total_soft_accuracy = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            if targets != "dynamic":
                target_vecs = generate_alt_target(target, targets).to(device)
            else:
                target_vecs = generate_target(target).to(device)

            output = network(data)
            probs = output.exp()
            
            if targets != "dynamic":
                loss = one_hot_nll_loss(output, target_vecs)
            else:
                loss = one_hot_nll_loss(output, apply_class_values(target_vecs, nc, c))
            test_loss += loss * data.size(0)

            ## Calculate cosine sim of closest class_target to output
            probs_norm = F.normalize(probs, dim=1)
            class_targets_norm = F.normalize(class_target_vectors, dim=1)

            # Cosine similarities: [B, N]
            similarities = torch.matmul(probs_norm, class_targets_norm.T)

            # Max similarity per row
            max_similarities, _ = similarities.max(dim=1)

            # Add to your total_confidence
            total_confidence += max_similarities.sum()

            ## Predicted class (based on nearest target vector)
            if targets == "dynamic":
                # print(probs)
                predicted = predict_by_nearest_target(probs, c, nc)
                total_correct += predicted.eq(target).sum().item()
            else:
                predicted = output.argmax(dim=1)
                total_correct += predicted.eq(target).sum().item()

            total_samples += data.size(0)

    test_loss /= total_samples
    avg_soft_acc = total_soft_accuracy / total_samples
    avg_conf = total_confidence / total_samples
    hard_acc = total_correct / total_samples
    log_data = {
        "hard accuracy": hard_acc,
        "confidence": avg_conf,
        "test loss": test_loss,
        "avg cosine similarity (confidence)": avg_conf,
        "epoch": epoch,
    }

    run.log(log_data)





###############################
##        RUN EXPERIMENT     ##
###############################

nc, c = pre_train(spacing=adaptation)
test(nc=nc, c=c, epoch=1, targets=targets)
for epoch in range(1, n_epochs + 1):
    nc, c = train(nc=nc, c=c, epoch=epoch, spacing=adaptation, targets=targets)
    test(epoch=float(epoch)+0.999, targets=targets, c=c, nc=nc)

run.finish()

