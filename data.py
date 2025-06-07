from torch.utils.data import DataLoader
import torchvision
import torch

def load_data(config):
    """takes an argparser instance, returns train and test loader"""
    train_loader: torch.utils.data.DataLoader[torchvision.datasets.MNIST] = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files/', train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,),(0.3081,))
            ])),
            batch_size=config.batch_size_train, shuffle=True)

    test_loader: torch.utils.data.DataLoader[torchvision.datasets.MNIST] = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files/', train=False, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,),(0.3081,))
            ])),
            batch_size=config.batch_size_train, shuffle=True)

    return train_loader, test_loader
