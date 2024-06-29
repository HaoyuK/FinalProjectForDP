import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from cutmix import *

def get_data_loader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    # Split the training data into training and validation sets
    val_size = len(cifar100_training) // 5
    train_size = len(cifar100_training) - val_size
    cifar100_train, cifar100_val = random_split(cifar100_training, [train_size, val_size])
    
    cifar100_train_loader = DataLoader(cifar100_train, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    cifar100_val_loader = DataLoader(cifar100_val, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    cifar100_test_loader = DataLoader(cifar100_test, shuffle=False, num_workers=num_workers, batch_size=batch_size)

    return cifar100_train_loader, cifar100_val_loader, cifar100_test_loader

def denormalize(tensor, mean, std):
    """反归一化操作，将归一化后的张量转换回原始范围."""
    if not torch.is_tensor(tensor):
        raise TypeError("Input should be a torch tensor.")
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    return tensor

def show_batch(images, labels, mean, std):
    import matplotlib
    matplotlib.use('TkAgg')
    images = denormalize(images, mean, std)
    img_grid = make_grid(images, nrow=4, padding=10, normalize=True)
    plt.imshow(img_grid.permute(1, 2, 0))
    plt.title(f"Labels: {labels}")
    plt.show()

if __name__ == "__main__":
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    mean = CIFAR100_TRAIN_MEAN
    std = CIFAR100_TRAIN_STD
    
    train_loader, val_loader, test_loader = get_data_loader(mean, std, batch_size=16, num_workers=2, shuffle=True)
    
    for images, labels in train_loader:
        print(images.shape)
        print(labels)
        break
