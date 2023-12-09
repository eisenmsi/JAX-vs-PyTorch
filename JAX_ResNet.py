import jax
from jax import numpy as jnp
from flax import linen as nn


class BasicBlock(nn.Module):
    expansion = 1

    @nn.compact
    def __call__(self, x, planes, stride=1):
        conv1 = nn.Conv(
            features=planes,
            kernel_size=(3, 3),
            strides=(stride, stride),
            padding=[(1, 1), (1, 1)],
            use_bias=False,
        )
        conv2 = nn.Conv(
            features=planes,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=[(1, 1), (1, 1)],
            use_bias=False,
        )

        shortcut = x
        if stride != 1 or x.shape[-1] != self.expansion * planes:
            shortcut = nn.Conv(
                features=self.expansion * planes,
                kernel_size=(1, 1),
                strides=(stride, stride),
                use_bias=False,
            )(x)

        y = conv1(x)
        y = nn.relu(y)
        y = conv2(y)

        y += shortcut
        y = nn.relu(y)
        return y


class ResNet18(nn.Module):
    num_blocks = [2, 2, 2, 2]  # Number of blocks in each stage for ResNet18

    @nn.compact
    def __call__(self, x, num_classes=10):
        conv1 = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=[(1, 1), (1, 1)],
            use_bias=False,
        )

        x = conv1(x)
        x = nn.relu(x)

        x = self._make_layer(x, 64, self.num_blocks[0], stride=1)
        x = self._make_layer(x, 128, self.num_blocks[1], stride=2)
        x = self._make_layer(x, 256, self.num_blocks[2], stride=2)
        x = self._make_layer(x, 512, self.num_blocks[3], stride=2)

        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        x = nn.Dense(num_classes)(x)
        return x

    def _make_layer(self, x, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            x = BasicBlock()(x, planes, stride)
        return x


if __name__ == "__main__":
    model = ResNet18()
    print(
        model.tabulate(
            jax.random.key(0),
            jnp.ones((1, 32, 32, 3)),
            compute_flops=True,
            compute_vjp_flops=True,
        )
    )
