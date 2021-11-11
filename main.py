import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from utils.viz import show_tensor_images

torch.manual_seed(0)  # Set for testing purposes, please do not change!
z_dim = 64
display_step = 500
batch_size = 128


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
])

dataloader = DataLoader(
    MNIST('.', download=False, transform=transform),batch_size=batch_size, shuffle=True)

real, label = next(iter(dataloader))
show_tensor_images(real)
