"""Generate ColorMNIST dataset.

Generate ColorMNIST dataset with different standard deviations of the color.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST


def generate_set(dataset, samples_per_class, train) -> TensorDataset:
    """Generate 30-class color MNIST dataset with long-tailed distribution."""

    # Convert dataset to numpy arrays.
    imgs, targets = dataset.data.numpy(), dataset.targets.numpy()
    print(f"Original dataset shape: {imgs.shape}, Target shape: {targets.shape}")

    if train:
        # Create power law distribution for 30 classes.
        samples = np.random.power(0.3, size=imgs.shape[0]) * samples_per_class
        samples = np.ceil(samples).astype(int)
        print(f"Train mode: Samples per class (power law): {samples[:10]}")
    else:
        # Create uniform distribution for 30 classes of 250 samples each.
        samples_per_class = 250
        samples = (np.ones(imgs.shape[0]) * samples_per_class).astype(int)
        print(f"Test mode: Samples per class (uniform): {samples[:10]}")

    # Convert grayscale images into a 30-class dataset (3 channels per digit)
    imgs_rgb = []
    targets_rgb = []

    for i in range(10):  # Iterate over the 10 digit classes
        samples_added = 0
        for j in range(3):  # Iterate over the 3 color channels
            class_idx = i * 3 + j  # Generate a new class index for each color variant
            print(f"Processing digit {i}, color channel {j} -> Class Index: {class_idx}")

            # Extract subset of images belonging to the current digit class
            data_tmp = imgs[targets == i][
                samples_added : samples_added + samples[class_idx]
            ]
            print(f"  Selected {data_tmp.shape[0]} samples for class {class_idx}")

            # Create an empty 3-channel array
            data = np.zeros(data_tmp.shape + (3,))
            data[:, :, :, j] = data_tmp  # Assign data to the j-th channel
            print("Data of shape:",data.shape)
            # Store processed images and targets
            imgs_rgb.append(data)
            targets_rgb.extend(list(np.ones(data.shape[0]) * class_idx))
            samples_added += samples[i]  # Update counter

    # Concatenate all images into a single array and normalize
    imgs_rgb = np.concatenate(imgs_rgb) / 255
    targets_rgb = np.asarray(targets_rgb)
    print(f"Final dataset size: {imgs_rgb.shape}, Targets: {targets_rgb.shape}")

    # Generate noisy background
    ims = imgs_rgb.shape[:3]  # Shape (N, H, W)
    weight = np.max(imgs_rgb, axis=3)  # Extract the max intensity per pixel
    noisy_background = args.bg_intensity + np.random.randn(*ims) * args.bg_noise_std
    noisy_background = np.clip(noisy_background, 0, 1)  # Ensure values are in [0,1]
    print(f"Noisy background shape: {noisy_background.shape}")

    # Blend images with the noisy background
    imgs_rgb = (
        weight[..., None] * imgs_rgb
        + (1 - weight[..., None]) * noisy_background[..., None]
    )

    # Convert to PyTorch tensors
    imgs_rgb = torch.from_numpy(imgs_rgb).permute(0, 3, 1, 2).float()  # (N, C, H, W)
    targets = torch.from_numpy(targets_rgb).long()
    print(f"Final tensor shapes -> Images: {imgs_rgb.shape}, Targets: {targets.shape}")

    return TensorDataset(imgs_rgb, targets)


def generate_colormnist_longtailed(samples_per_class):
    # Create out directory.
    os.makedirs(os.environ["DATA_DIR"] + "/colormnist_longtailed", exist_ok=True)

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

    trainset = generate_set(trainset, samples_per_class, True)
    trainset_gray = TensorDataset(
        trainset.tensors[0].mean(1, keepdim=True), trainset.tensors[1]
    )
    testset = generate_set(testset, samples_per_class, False)

    # Save imagegrid.
    grid_img = torchvision.utils.make_grid(trainset.tensors[0], nrow=32)
    plt.rcParams.update({"font.size": 7})
    plt.figure(figsize=(8, 16))
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    _ = plt.xticks([]), plt.yticks([])
    plt.savefig(
        os.environ["DATA_DIR"] + "/colormnist_longtailed/colormnist_longtailed.png",
        bbox_inches="tight",
    )
    plt.clf()

    grid_img = torchvision.utils.make_grid(trainset_gray.tensors[0], nrow=32)
    plt.rcParams.update({"font.size": 7})
    plt.figure(figsize=(8, 16))
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    _ = plt.xticks([]), plt.yticks([])
    plt.savefig(
        os.environ["DATA_DIR"]
        + "/colormnist_longtailed/colormnist_longtailed_gray.png",
        bbox_inches="tight",
    )
    plt.clf()

    # Save datasets.
    torch.save(
        trainset,
        os.environ["DATA_DIR"] + "/colormnist_longtailed/train.pt",
    )
    torch.save(
        testset,
        os.environ["DATA_DIR"] + "/colormnist_longtailed/test.pt",
    )

    print(
        "Generated ColorMNIST - longtailed dataset at {}".format(
            os.environ["DATA_DIR"] + "/colormnist_longtailed"
        )
    )

    # Plot and save histogram of samples per class.
    samples_per_class = torch.unique(trainset.tensors[1], return_counts=True)
    sort_idx = torch.argsort(samples_per_class[1], descending=True)
    samples_per_class = (samples_per_class[0][sort_idx], samples_per_class[1][sort_idx])

    labels = [j + str(i) for i in range(10) for j in ["R", "G", "B"]]
    labels = [labels[i] for i in sort_idx.numpy()]

    plt.figure(figsize=(6, 3))
    plt.bar(labels, samples_per_class[1].numpy())
    plt.savefig(
        os.environ["DATA_DIR"] + "/colormnist_longtailed/histogram.png",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    """Generate dataset."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples_per_class", type=int, default=150, help="Samples per class."
    )
    parser.add_argument(
        "--bg_noise_std", type=float, default=0.1, help="std of background noise"
    )
    parser.add_argument(
        "--bg_intensity", type=float, default=0.33, help="intensity of background"
    )
    args = parser.parse_args()

    generate_colormnist_longtailed(args.samples_per_class)
