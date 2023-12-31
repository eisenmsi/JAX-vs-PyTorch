# JAX vs. PyTorch

This repository contains implementations of a ResNet model for the CIFAR-10 dataset using both PyTorch and JAX/Flax/Optax frameworks. It compares the performance and training times of these implementations.

## Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal is to classify these images into their respective categories.

### Implementations

1. **PyTorch Implementation:** This implementation utilizes PyTorch for building and training the ResNet model for CIFAR-10.

2. **JAX/Flax/Optax Implementation:** This implementation uses JAX with Flax and Optax libraries to create and train the ResNet model for CIFAR-10.

## Prerequisites

Ensure you have the following installed:

- Python (preferably Python 3.x)
- PyTorch (for the PyTorch implementation)
- JAX, Flax, and Optax (for the JAX implementation)

## Metrics

- **Accuracy:** Compare the accuracy achieved by both implementations.
- **Training Time:** Measure and compare the time taken by each implementation to train the model.

## Notes

- Experiment with different hyperparameters, architectures, or optimizers to further explore the differences between PyTorch and JAX implementations.
- Consider adjusting the number of epochs based on your machine's capabilities for a more comprehensive comparison.

## Frameworks Comparison: PyTorch vs. JAX/Flax/Optax

Understanding the differences between PyTorch and JAX/Flax/Optax is crucial for selecting the appropriate framework. Here's a comparative analysis:

### PyTorch

- **Programming Paradigm:** Primarily imperative, supports dynamic computation graphs.
- **Automatic Differentiation:** Provides automatic differentiation through `autograd`.
- **Ecosystem and Community:** Large community, extensive documentation, rich ecosystem.
- **Model Building and Optimization:** High-level API simplifies model building and optimization.
- **Performance:** Flexible and easy to use, slightly less optimized for specific scenarios.

### JAX/Flax/Optax

- **Programming Paradigm:** Functional-first, composable transformations.
- **Automatic Differentiation:** Offers automatic differentiation through `jax.grad`.
- **Ecosystem and Community:** Growing popularity in research, Flax and Optax provide additional functionalities.
- **Model Building and Optimization:** Lower-level primitives, requires more explicit architecture definitions.
- **Performance:** Designed for high-performance computations, especially on accelerators.

## How to Choose?

The choice between PyTorch and JAX/Flax/Optax depends on various factors:

- **Project Requirements:** Consider the project's specific needs regarding performance, ease of use, and ecosystem support.
- **Familiarity:** Your familiarity with either framework can influence the choice.
- **Use Case:** Whether the project is for research, production, or performance-critical applications.

You can choose the framework that best aligns with the project's goals and your expertise.

## Contributions

Contributions are welcome! Feel free to open issues or pull requests for improvements, bug fixes, or additional experiments.

