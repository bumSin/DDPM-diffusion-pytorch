import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose

def prepare_data_loader(batch_size = 128):
    transform = Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # Number of images per batch
        shuffle=True  # Shuffle the data at every epoch
    )

    return dataloader