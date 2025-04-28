import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST

def generate_luminance_set(dataset, samples_per_class, train) -> TensorDataset:
    """Generate a luminance MNIST dataset with high, medium, and low brightness levels."""
    imgs, targets = dataset.data.numpy(), dataset.targets.numpy()
    
    if train:
        # Create power law distribution for 10 classes.
        samples = np.random.power(0.3, size=imgs.shape[0]) * samples_per_class
        samples = np.ceil(samples).astype(int)
    else:
        # Create uniform distribution for 10 classes of 250 samples each.
        samples_per_class = 250
        samples = (np.ones(imgs.shape[0]) * samples_per_class).astype(int)
    
    imgs_gray = []
    targets_gray = []
    brightness_levels = [0.5, 1.5]  # Low, High luminance
    
    for i in range(10):
        samples_added = 0
        for j, brightness_factor in enumerate(brightness_levels):
            class_idx = i * 2 + j
            
            # Get data.
            data_tmp = imgs[targets == i][
                samples_added : samples_added + samples[class_idx]
            ]
            
            # Normalize grayscale images to range [0,1]
            data = data_tmp.astype(np.float32) / 255.0
            
            # Apply predefined brightness variations
            data = np.clip(data * brightness_factor, 0, 1)
            
            # Add data to list.
            imgs_gray.append(data[:, :, :, None])  # Keep single channel
            targets_gray.extend(list(np.ones(data.shape[0]) * class_idx))
            samples_added += samples[i]
    
    # Concatenate samples and targets.
    imgs_gray = np.concatenate(imgs_gray, axis=0)
    targets_gray = np.asarray(targets_gray)
    
    # Convert to tensor.
    imgs_gray = torch.from_numpy(imgs_gray).permute(0, 3, 1, 2).float()
    targets_gray = torch.from_numpy(targets_gray).long()
    
    return TensorDataset(imgs_gray, targets_gray)


def generate_luminance_mnist(samples_per_class):
    """Generate LuminanceMNIST dataset."""
    os.makedirs(os.environ["DATA_DIR"] + "/luminance_mnist", exist_ok=True)
    
    trainset = MNIST(
        root=os.environ["DATA_DIR"] + "/MNIST",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    testset = MNIST(
        root=os.environ["DATA_DIR"] + "/MNIST",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    
    trainset = generate_luminance_set(trainset, samples_per_class, True)
    testset = generate_luminance_set(testset, samples_per_class, False)
    
    # Save image grid.
    grid_img = torchvision.utils.make_grid(trainset.tensors[0], nrow=32)
    plt.figure(figsize=(8, 16))
    plt.imshow(grid_img.permute(1, 2, 0).numpy(), cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.savefig(
        os.environ["DATA_DIR"] + "/luminance_mnist/luminance_mnist.png",
        bbox_inches="tight",
    )
    plt.clf()
    
    # Save datasets.
    torch.save(trainset, os.environ["DATA_DIR"] + "/luminance_mnist/train.pt")
    torch.save(testset, os.environ["DATA_DIR"] + "/luminance_mnist/test.pt")
    
    print(f"Generated LuminanceMNIST dataset at {os.environ['DATA_DIR']}/luminance_mnist")
    
    # Plot and save histogram of samples per class.
    samples_per_class = torch.unique(trainset.tensors[1], return_counts=True)
    sort_idx = torch.argsort(samples_per_class[1], descending=True)
    samples_per_class = (samples_per_class[0][sort_idx], samples_per_class[1][sort_idx])
    
    labels = [str(i) for i in range(30)]
    labels = [labels[i] for i in sort_idx.numpy()]
    
    plt.figure(figsize=(6, 3))
    plt.bar(labels, samples_per_class[1].numpy())
    plt.savefig(
        os.environ["DATA_DIR"] + "/luminance_mnist/histogram.png",
        bbox_inches="tight",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_per_class", type=int, default=150, help="Samples per class.")
    args = parser.parse_args()
    generate_luminance_mnist(args.samples_per_class)
