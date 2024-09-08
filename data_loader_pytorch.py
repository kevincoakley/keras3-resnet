import torch
import torchvision
import random
import numpy as np


# Custom transform to permute dimensions for use with Keras 3 models
class PermuteTransform:
    def __call__(self, x):
        return x.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)


def load_dataset(dataset_details, batch_size, seed_val=0):

    # Set seed for reproducibility
    if seed_val != 0:
        torch.manual_seed(seed_val)
        random.seed(seed_val)
        np.random.seed(seed_val)

    # Get the dataset details
    train_path = dataset_details["train_path"]
    val_path = dataset_details["val_path"]
    test_path = dataset_details["test_path"]
    dataset_shape = dataset_details["dataset_shape"]
    normalization_mean = dataset_details["normalization"]["mean"]
    normalization_std = dataset_details["normalization"]["std"]

    normalize = torchvision.transforms.Normalize(
        mean=normalization_mean, std=normalization_std
    )

    #
    # Define the preprocessing and augmentation transformations
    #
    preprocessing = torchvision.transforms.Compose(
        [
            # Resize the image to the dataset shape
            torchvision.transforms.Resize(
                (dataset_shape[0], dataset_shape[1]),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                antialias=False,
            ),
            torchvision.transforms.ToTensor(),
            # Normalize the pixel values ((input[channel] - mean[channel]) / std[channel])
            normalize,
            # Permute dimensions for use with Keras 3 model
            PermuteTransform(),
        ]
    )

    preprocessing_augument = torchvision.transforms.Compose(
        [
            # Resize the image to the dataset shape
            torchvision.transforms.Resize(
                (dataset_shape[0], dataset_shape[1]),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                antialias=False,
            ),
            # Randomly pad the image by 25% of the dataset shape
            torchvision.transforms.Pad(
                (round(dataset_shape[0] * 0.25), round(dataset_shape[1] * 0.25))
            ),
            # Randomly crop the image to the dataset shape
            torchvision.transforms.RandomCrop((dataset_shape[0], dataset_shape[1])),
            # Randomly flip the image horizontally
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            # Normalize the pixel values ((input[channel] - mean[channel]) / std[channel])
            normalize,
            # Permute dimensions for use with Keras 3 model
            PermuteTransform(),
        ]
    )

    #
    # Get the training, validation, and test datasets from the directory
    #
    train = torchvision.datasets.ImageFolder(
        root=train_path, transform=preprocessing_augument
    )
    val = torchvision.datasets.ImageFolder(root=val_path, transform=preprocessing)
    test = torchvision.datasets.ImageFolder(root=test_path, transform=preprocessing)

    print(f"Number of training samples: {len(train)}")
    print(f"Number of validation samples: {len(val)}")
    print(f"Number of test samples: {len(test)}")

    #
    # Create the data loaders
    #
    train_dataset = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_dataset = torch.utils.data.DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_dataset = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_dataset, val_dataset, test_dataset
