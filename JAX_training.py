import jax
from jax import numpy as jnp
import optax
import time
import torch
import torchvision
import torchvision.transforms as transforms
import wandb
from JAX_ResNet import ResNet18  # Import ResNet18
from jax import device_put

from jax.lib import xla_bridge

print(xla_bridge.get_backend().platform)  # Check if GPU is used


def loss_fn(params, inputs, targets, model):
    logits = model.apply({"params": params}, inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    return loss


def update(params, inputs, targets, optimizer, opt_state, model):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(params, inputs, targets, model)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def train(params, dataloader, opt_state, model, optimizer):
    for batch, (X, y) in enumerate(dataloader):
        X = jnp.transpose(X, axes=(0, 2, 3, 1))
        params, opt_state = update(params, X, y, optimizer, opt_state, model)
    return params, opt_state


def test(params, dataloader, model):
    total_loss = 0.0

    for batch, (X, y) in enumerate(dataloader):
        X = jnp.transpose(X, axes=(0, 2, 3, 1))
        total_loss += loss_fn(params, X, y, model)

    test_loss = total_loss / len(dataloader)

    return test_loss


if __name__ == "__main__":
    wandb.init(project="Flax Training")

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

    train_ds, val_ds = torch.utils.data.random_split(trainset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Convert to numpy arrays
    train_numpy = [(jnp.array(x), jnp.array(y)) for x, y in train_loader]
    val_numpy = [(jnp.array(x), jnp.array(y)) for x, y in val_loader]
    test_numpy = [(jnp.array(x), jnp.array(y)) for x, y in test_loader]

    # Move to GPU
    train_numpy = device_put(train_numpy)
    val_numpy = device_put(val_numpy)
    test_numpy = device_put(test_numpy)

    model = ResNet18()

    lr = 0.001  # Initial learning rate

    # Initialize the model's parameters and the optimizer
    variables = jax.jit(model.init)(
        jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 3))
    )  # initialize variables by passing a template image
    params = variables["params"]

    # Initialize the optimizer
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    total_iterations = 20  # Number of iterations
    iterations = 0
    best_val_loss = float("inf")

    lr_decay_factor = 0.1
    patience = 10
    no_improvement_count = 0

    val_loss = test(params, val_numpy, model)

    # Log metrics to wandb
    wandb.log({"Validation loss": val_loss})

    start = time.process_time()
    start_2 = time.time()
    while iterations < total_iterations:
        params, opt_state = train(params, train_numpy, opt_state, model, optimizer)
        val_loss = test(params, val_numpy, model)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            lr *= lr_decay_factor
            optimizer = optax.adam(learning_rate=lr)
            opt_state = optimizer.init(params)
            no_improvement_count = 0

        iterations += 1

        # Log metrics to wandb
        wandb.log({"Validation loss": val_loss})

    end = time.process_time()
    end_2 = time.time()

    print("Best validation loss: " + str(best_val_loss))
    print("Process time: " + str(end - start) + "s")
    print("Wall-clock time: " + str(end_2 - start_2) + "s")
