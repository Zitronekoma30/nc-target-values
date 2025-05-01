# Initial Code: https://nextjournal.com/gkoehler/pytorch-mnist
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Dict, Tuple



# ---------------------------------------------- #
#               SETUP / LOADING DATA             #
# ---------------------------------------------- #

# set values
n_epochs = 4
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1

CLASSES = 10

torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# load data

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,),(0.3081,))
        ])),
        batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
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

#######################################
##          SPACING METHODS          ##
#######################################

def base_method(class_targets: Dict[Tuple, float], nclass_target: float, outputs):
    """Serves as stand in if no spacing is desired, returns back existing nc and c"""
    return  nclass_target, class_targets

def sigma_method(class_targets: Dict[Tuple, float], nclass_target: float, outputs):
    """Spaces the class/non class values using a combination of their std deviation, returns new nc and c"""
    std_devs_class = {}
    std_dev_nc = 0
    # REMEMBER .exp() to get to probability space for std dev
    # .log() to go back to logs
    # use probs for calculation use logs for training
    new_class, new_nclass = 0, 0
    return new_nclass, new_class

##############################################
##          TRAINING AND TESTING            ##
##############################################

def train(epoch: int, nc: float, c: Dict[Tuple, float], spacing: Callable = base_method):
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
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader)*batch_size_train} Loss: {loss.item()}]")

            train_losses.append(loss.item())
            train_counter.append((epoch - 1) * len(train_loader.dataset) + batch_idx * len(data))

            torch.save(network.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')

    return spacing(class_targets, nclass_target, outputs)

def test():
    """Tests network using test dataset"""
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = network(data)
            # test_loss += F.nll_loss(output, target, size_average=False).item()
            loss = one_hot_nll_loss(output, generate_target(target).to(device))
            test_loss += loss * data.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.to(device).data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



###############################
##        RUN EXPERIMENT     ##
###############################

nc, c = pre_train()
test()
for epoch in range(1, n_epochs + 1):
  train(epoch, nc=nc, c=c)
  test()

fig = plt.figure()
plt.plot(
    [x.item() if torch.is_tensor(x) else x for x in train_counter],
    [y.item() if torch.is_tensor(y) else y for y in train_losses],
    color='blue'
)
plt.scatter(
    [x.item() if torch.is_tensor(x) else x for x in test_counter],
    [y.item() if torch.is_tensor(y) else y for y in test_losses],
    color='red'
)
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig.show()
input()
