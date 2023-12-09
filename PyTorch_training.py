import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import time
from torch.utils.data import random_split
from PyTorch_ResNet import ResNet18  # Import ResNet18
import wandb


def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for batch, data in enumerate(dataloader):
        X = data[0]
        y = data[1]
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, model, loss_fn, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in dataloader:
            X = data[0]
            y = data[1]
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= len(dataloader)

    return test_loss


if __name__ == "__main__":
    wandb.init(
        project="PyTorch Training",
    )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 32

    torch.manual_seed(42)
    val_size = 5000

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    train_size = len(trainset) - val_size

    train_ds, val_ds = random_split(trainset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size, num_workers=4, pin_memory=True
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18().to(device)  # Modify parameters as needed

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999)
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )

    total_iterations = 20  # number of iterations
    iterations = 0
    best_val_loss = float("inf")

    val_loss = test(val_loader, model, loss_fn, device)

    # log metrics to wandb
    wandb.log({"Validation loss": val_loss})

    start = time.process_time()
    start_2 = time.time()
    while iterations < total_iterations:
        model.train()
        train(train_loader, model, loss_fn, optimizer, device)
        val_loss = test(val_loader, model, loss_fn, device)
        lr_scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        iterations += 1

        # log metrics to wandb
        wandb.log({"Validation loss": val_loss})

    end = time.process_time()
    end_2 = time.time()

    print("Best validation loss: " + str(best_val_loss))
    print("Process time: " + str(end - start) + "s")
    print("Wall-clock time: " + str(end_2 - start_2) + "s")
