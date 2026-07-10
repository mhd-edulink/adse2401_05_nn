# -----------------------------------------------------------------------
# 0. Import required modules
# -----------------------------------------------------------------------
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import random
import torch

from scipy.ndimage import rotate
from torch import nn
from torch._C import device
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

import warnings

# Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Constants
# -----------------------------------------------------------------------------
IMAGE_SIZE = 32
NUM_CLASSES = 3
TRAIN_SAMPLES = 1200
TEST_SAMPLES = 300
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 0.001

# Optional if you have a cude enabled GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# 2. Other Constants For Reproducibile Results
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# -----------------------------------------------------------------------------
# 3. Synthetic dataset
# -----------------------------------------------------------------------------
class ShapeDataset(Dataset):
    """
    Synthetic dataset of simple geometric shapes.

    The dataset is generated entirely in memory. Each image contains one
    randomly generated geometric shape belonging to one of three
    classes:

        * Square
        * Circle
        * Triangle

    Several random variations are introduced to make the classification
    problem more realistic:

    * Background noise
    * Variable brightness
    * Variable contrast
    * Random object size
    * Random object position
    * Filled or hollow shapes
    * Random rotation (triangles only)
    * Random image occlusion

    Parameters
    ----------
    number_of_samples : int
        Number of images to generate.
    """

    def __init__(self, number_of_samples: int) -> None:

        self.images = []
        self.labels = []

        for _ in range(number_of_samples):
            image = self._create_background()

            label = random.randint(0, 2)

            if label == 0:
                self._draw_square(image)

            elif label == 1:
                self._draw_circle(image)

            else:
                self._draw_triangle(image)

            self._apply_occlusion(image)
            image = np.clip(image, 0, 1.0)

            self.images.append(image.astype(np.float32))
            self.labels.append(label)

    # -----------------------------------------------------------------------------
    # Dataset interference
    # -----------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return one image and its label.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        tuple
            Image tensor and corresponding class label.
        """
        image = torch.tensor(self.images[index], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[index], dtype=torch.long)

        return image, label

    # -----------------------------------------------------------------------------
    # Image generation utilities
    # -----------------------------------------------------------------------------
    @staticmethod
    def _create_background() -> np.ndarray:
        background = random.uniform(0.05, 0.30)

        image = np.random.normal(
            loc=background,
            scale=0.05,
            size=(IMAGE_SIZE, IMAGE_SIZE)
        )

        return image

    @staticmethod
    def _random_brightness(background: float) -> float:
        low = min(background + 0.25, 0.9)
        return random.uniform(low, 1.0)

    @staticmethod
    def _random_position() -> Tuple[int, int]:
        x = random.randint(4, 18)
        y = random.randint(4, 18)
        return x, y

    @staticmethod
    def _random_size() -> int:
        return random.randint(8, 16)

    @staticmethod
    def _apply_occlusion(image: np.ndarray) -> None:
        if random.random() > 0.3:
            return

        x = random.randint(0, IMAGE_SIZE - 8)
        y = random.randint(0, IMAGE_SIZE - 8)

        width = random.randint(4, 8)
        height = random.randint(4, 8)

        image[y:y + height, x:x + width] = np.mean(image)

    # -----------------------------------------------------------------------------
    # Shape Drawing methods
    # -----------------------------------------------------------------------------
    def _draw_square(self, image: np.ndarray) -> None:
        """
        Draw a square.

        Squares may be either filled or hollow. Hollow squares are drawn
        using a randomly selected border thickness.
        """
        x, y = self._random_position()
        size = self._random_size()
        background = float(np.mean(image))
        brightness = self._random_brightness(background)
        filled = random.choice([True, False])

        if filled:
            image[y:y + size, x:x + size] = brightness
            return

        thickness = random.randint(1, 4)

        # Top border
        image[y:y + thickness, x:x + size] = brightness

        # Bottom border
        image[y + size - thickness:y + size, x:x + size] = brightness

        # Left border
        image[y:y + size, x:x + thickness] = brightness

    def _draw_circle(self, image: np.ndarray) -> None:
        """
        Draw a circle.

        Circles may be either filled or hollow. Hollow circle are drawn
        using a randomly selected border thickness.
        """
        radius = random.randint(4, 8)
        cx = random.randint(radius + 2, IMAGE_SIZE - radius - 2)
        cy = random.randint(radius + 2, IMAGE_SIZE - radius - 2)
        background = float(np.mean(image))
        brightness = self._random_brightness(background)

        yy, xx = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
        distance = (
                (xx - cx) ** 2 + (yy - cy) ** 2
        )

        filled = random.choice([True, False])
        if filled:
            mask = distance <= radius ** 2
        else:
            thickness = random.randint(1, 2)
            outer = distance <= radius ** 2
            inner = distance <= (
                    radius - thickness
            ) ** 2
            mask = outer & (~inner)

        image[mask] = brightness

    def _draw_triangle(self, image: np.ndarray) -> None:
        """
        Draw a randomly rotated triangle.

        A simple upright triangle is first generated and is then rotated
        through a random angle. The rotated triangle is finally copied
        onto the image.

        This approach is computationally inexpensive while producing a
        much richer variety of training examples.
        """
        size = self._random_size()
        background = float(np.mean(image))
        brightness = self._random_brightness(background)
        filled = random.choice([True, False])
        triangle = np.zeros(
            (size, size), dtype=np.float32
        )

        centre = size // 2

        # -----------------------------------------------------------------------------
        # Draw an upright triangle
        # -----------------------------------------------------------------------------
        for row in range(size):
            half_width = int(row * size / (2 * size))

            left = max(centre - half_width, 0)

            right = min(centre + half_width, 1, size)

            if filled:
                triangle[
                    row, left:right
                ] = brightness

            else:
                triangle[
                    row, left
                ] = brightness

                triangle[
                    row, right - 1
                ] = brightness

        if not filled:
            triangle[
                size - 1, :
            ] = brightness

        # -----------------------------------------------------------------------------
        # Apply a random rotation
        # -----------------------------------------------------------------------------
        angle = random.uniform(-45.0, 45.0)
        triangle = rotate(triangle, angle=angle, reshape=False, order=1, mode='constant', cval=0.0)

        # -----------------------------------------------------------------------------
        # Copy the rotated triangle into the image.
        # -----------------------------------------------------------------------------
        x = random.randint(0, IMAGE_SIZE - size)
        y = random.randint(0, IMAGE_SIZE - size)
        mask = triangle > 0.05
        region = image[y:y + size, x:x + size]
        region[mask] = triangle[mask]


# -----------------------------------------------------------------------------
# 4. Residual Block Class
# -----------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    Basic residual block.

    The block learns a residual mapping F(x), which is added to the
    original input before the activation function is applied.

        Output = ReLU(F(x) + x)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(x)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out


# -----------------------------------------------------------------------------
# 5. Simple Residual Network(resnet)
# -----------------------------------------------------------------------------
# class SimpleResNet(nn.Module):
#     """
#     A small Residual Network for classroom demonstrations.
#
#     The network is deliberately compact so that students can easily
#     understand the flow of data through convolutional, residual and
#     classification layers.
#     """
#
#     def __init__(self) -> None:
#         super().__init__()
#
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             ResidualBlock(16),
#             ResidualBlock(16),
#             nn.MaxPool2d(2),
#             nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             ResidualBlock(32),
#             ResidualBlock(32),
#             nn.MaxPool2d(2),
#         )
#
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(p=0.3),
#             nn.Linear(32, 64),
#             nn.ReLU(inplace=True),
#             nn.Linear(64, NUM_CLASSES),
#         )
#
#         def forward(self, x: torch.Tensor) -> torch.Tensor:
#             x = self.features(x)
#             x = self.pool(x)
#             x = self.classifier(x)
#             return x
class SimpleResNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ResidualBlock(16),
            ResidualBlock(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            ResidualBlock(32),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):      # ✅ Same indentation as __init__
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

# -----------------------------------------------------------------------------
# 6. Function to train a model for an epoch
# -----------------------------------------------------------------------------
def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
) -> tuple[float, float]:
    """
    Train a model for one epoch.

    Parameters
    ----------
    model : nn.Module
        Neural network to train.
    loader : DataLoader
        Training data.
    criterion : nn.Module
        Loss function.
    optimiser : torch.optim.Optimizer
        Optimiser.

    Returns
    -------
    tuple
        Average training loss and training accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()

    average_loss = running_loss / len(loader)
    accuracy = correct / len(loader)
    return average_loss, accuracy


# -----------------------------------------------------------------------------
# 7. Evaluation
# -----------------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        average_loss = running_loss / len(loader)
        accuracy = correct / total
        return average_loss, accuracy


# -----------------------------------------------------------------------------
# 8. Model Training
# -----------------------------------------------------------------------------
def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, model_name: str) -> dict[
    str, list[float]]:
    criterion = nn.CrossEntropyLoss()

    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    print(f"n{'=' * 60}")
    print(f"Training {model_name}")
    print(f"n{'=' * 60}")

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimiser)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        print(
            f"Epoch {epoch + 1:2d}/{EPOCHS} "
            f"Train Loss {train_loss:.4f}  "
            f"Train Accuracy {train_accuracy:.4f}  "
            f"Test Loss {train_accuracy:.4f}  "
            f"Test Accuracy {train_accuracy:.4f}  "
        )

    return history


# -----------------------------------------------------------------------------
# 9. Visualisation
# -----------------------------------------------------------------------------
def plot_history(history: dict[str, list[float]], title: str) -> None:
    """
    Plot loss and accuracy curves for a trained model
    :param history:
    :param title:
    :return:
    """

    epochs = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # -----------------------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------------------
    axes[0].plot(epochs, history["train_loss"], label="Training Loss", marker="o")
    axes[0].plot(epochs, history["test_loss"], label="Test Loss", marker="s")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True)
    axes[0].legend()

    # -----------------------------------------------------------------------------
    # Accuracy
    # -----------------------------------------------------------------------------
    axes[0].plot(epochs, history["train_accuracy"], label="Training Accuracy", marker="o")
    axes[0].plot(epochs, history["test_accuracy"], label="Testing Accuracy", marker="s")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True)
    axes[0].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 10. Dataset Visualisation
# -----------------------------------------------------------------------------
def show_dataset_examples(dataset: ShapeDataset, rows: int = 2, columns: int = 6) -> None:
    class_names = ["Square", "Circle", "Triangle"]
    figure, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(12, 5))
    axes = axes.flatten()

    for axis in axes:
        index = random.randint(0, len(dataset) - 1)
        image, label = dataset[index]
        axis.imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
        axis.set_title(class_names[label.item()], fontsize=10)
        axis.axis("off")

    figure.suptitle("Examples from the Synthetic Dataset", fontsize=15)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 11. Results Summary
# -----------------------------------------------------------------------------
def print_summary(resnet_history: dict[str, list[float]]) -> None:
    resnet_accuracy = resnet_history["test_accuracy"][-1]
    print(f"\n{'=' * 60}")
    print(f"Final Results")
    print(f"{'=' * 60}")
    print(f"\n{'=' * 60}")
    print(f"ResNet Final Accuracy: {resnet_accuracy:.3f}")


# -----------------------------------------------------------------------------
# 12. Main Execution Method
# -----------------------------------------------------------------------------
def main() -> None:
    print("\nGENERATING SYNTEHTIC DATASET...\n")
    train_dataset = ShapeDataset(TRAIN_SAMPLES)
    test_dataset = ShapeDataset(TEST_SAMPLES)
    show_dataset_examples(train_dataset, rows=2, columns=2)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    resnet = SimpleResNet().to(DEVICE)
    resnet_history = train_model(resnet, train_loader, test_loader, model_name="Residual Network")
    print_summary(resnet_history)


# -----------------------------------------------------------------------------
# 13. Run the script by invokin its main function
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
