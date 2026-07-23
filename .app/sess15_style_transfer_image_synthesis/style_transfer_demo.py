"""
Python script to demonstrate Conditional Deep Convolutional GAN (cDCGAN) clothin synthesis demonstration
This script demonstrates CONDITIONAL IMAGE SYNTHESIS
using a Deep Convolutional Generative Adversarial Network (DCGAN),
conditioned on clothing category, trained on Fashion-MNIST. Given a
requested category label (for example, "Sneaker" or "Dress"), the trained
GENERATOR produces new, synthetic clothing images belonging to that
category.

On first run, the script will:

    1. Download the Fashion-MNIST dataset (cached for subsequent runs;
       shared with 'cvae_fashion_mnist_demo.py' if run from the same
       directory).
    2. Train a small conditional DCGAN.
    3. Save the trained generator to 'output/cgan_fashion_mnist.pth'.
    4. Generate a labelled grid of newly synthesised clothing images for
       every category and save it to
       'output/generated_clothing_grid_gan.png'.

On subsequent runs, if a saved checkpoint is found, training is skipped
and the script goes straight to generating a fresh grid of images.

Typical running time: several minutes on an ordinary CPU. This is
noticeably slower than the CVAE demonstration, because every training
step requires TWO forward-and-backward passes (one for the discriminator,
one for the generator) through convolutional networks, rather than one
pass through fully connected layers. This difference in training cost is
itself a useful discussion point when comparing the two approaches.

Dependencies
------------
- torch          (PyTorch; model definition, training and inference)
- torchvision    (provides automatic access to the Fashion-MNIST dataset)
- opencv-python  (cv2; used to build and save the output image grid)
- numpy          (array manipulation)
- pathlib        (filesystem paths; standard library)
- typing         (type hints; standard library)

Requirements:
    !pip install torch torchvision opencv-python numpy
"""

# -------------------------------------------------------------------------------------------------
# 0. Import required modules
# -------------------------------------------------------------------------------------------------
from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as functional
import torchvision

from pathlib import Path
from torch import nn, Tensor
from torch.utils.data import DataLoader
from typing import final

import warnings
# Surpress warning for cleaner output demo
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------------------------
# 1. Constants
# -------------------------------------------------------------------------------------------------
IMAGE_HEIGHT: Final[int] = 28
IMAGE_WIDTH: Final[int] = 28

NUMBER_OF_CLASSES: Final[int] = 10
CLASS_NAMES: Final[list[str]] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
"""The official Fashion-MNIST class names, in label order (0 to 9)."""

LATENT_DIMENSIONALITY: Final[int] = 100
"""The size of the generator's input noise vector. This is unrelated to,
and considerably larger than, the CVAE's latent space, which is
conventional for GANs since the generator must map noise directly to
pixels with no encoder to guide it."""

GENERATOR_INITIAL_SPATIAL_SIZE: Final[int] = 7
"""The generator first projects its input into a 7 x 7 feature map, which
is then upsampled (7 -> 14 -> 28) to reach the final image resolution."""

GENERATOR_INITIAL_CHANNELS: Final[int] = 128
DISCRIMINATOR_BASE_CHANNELS: Final[int] = 64

NUMBER_OF_TRAINING_EPOCHS: Final[int] = 12
BATCH_SIZE: Final[int] = 256
LEARNING_RATE: Final[float] = 2e-4
"""A learning rate of 2e-4 is a widely used, empirically stable default
for DCGAN training, originating from the original DCGAN paper."""

ADAM_BETA_1: Final[float] = 0.5
"""Reducing Adam's first momentum coefficient from its usual default of
0.9 to 0.5 is another well-known DCGAN stability trick, which reduces
oscillation during the adversarial training process."""

REAL_LABEL_SMOOTHING_VALUE: Final[float] = 0.9
"""Rather than training the discriminator to output exactly 1.0 for real
images, we train it towards 0.9. This "label smoothing" trick makes the
discriminator less overconfident, which in turn provides steadier
gradients to the generator."""

SAMPLES_PER_CLASS: Final[int] = 6
"""How many newly generated images to show per clothing category in the
output grid."""

DISPLAY_TILE_SIZE: Final[int] = 96
"""Each generated 28 x 28 image is up-scaled to this size (in pixels) so
that it is clearly visible when projected during a lecture."""

LABEL_MARGIN_WIDTH: Final[int] = 170
"""Width, in pixels, reserved on the left of the output grid for printing
each category's name."""

SCRIPT_DIRECTORY: Final[Path] = Path(__file__).resolve().parent
DATA_DIRECTORY: Final[Path] = SCRIPT_DIRECTORY / "data"
OUTPUT_DIRECTORY: Final[Path] = SCRIPT_DIRECTORY / "output"
MODEL_CHECKPOINT_PATH: Final[Path] = OUTPUT_DIRECTORY / "cgan_fashion_mnist.pth"
GENERATED_GRID_PATH: Final[Path] = OUTPUT_DIRECTORY / "generated_clothing_grid_gan.png"


# -------------------------------------------------------------------------------------------------
# 2. Model Architecture
# -------------------------------------------------------------------------------------------------
class Generator(nn.Module):
    """Generates a synthetic clothing image from noise and a class label.

    Educational note: the generator never observes a real image during
    training. It only ever receives random noise (the latent vector) and
    a class label, and learns purely from the discriminator's feedback.
    """

    def __init__(self) -> None:
        """Build the projection and upsampling layers of the generator."""
        super().__init__()
        input_size = LATENT_DIMENSIONALITY + NUMBER_OF_CLASSES
        projected_size = (
                GENERATOR_INITIAL_CHANNELS
                * GENERATOR_INITIAL_SPATIAL_SIZE
                * GENERATOR_INITIAL_SPATIAL_SIZE
        )

        self.projection_layer = nn.Sequential(
            nn.Linear(input_size, projected_size),
            nn.BatchNorm1d(projected_size),
            nn.ReLU(inplace=True),
        )

        self.upsampling_layers = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(
                GENERATOR_INITIAL_CHANNELS, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            # Tanh squashes pixel values into [-1, 1], matching the
            # normalisation applied to the real training images.
            nn.Tanh(),
        )

    def forward(self, noise_vectors: Tensor, one_hot_labels: Tensor) -> Tensor:
        """Generate a batch of images.

        Parameters
        ----------
        noise_vectors:
            A tensor of shape ``(batch_size, LATENT_DIMENSIONALITY)``
            sampled from a standard Normal distribution.
        one_hot_labels:
            A tensor of shape ``(batch_size, NUMBER_OF_CLASSES)``.

        Returns
        -------
        Tensor
            A tensor of shape ``(batch_size, 1, IMAGE_HEIGHT,
            IMAGE_WIDTH)`` with pixel values in ``[-1, 1]``.
        """
        combined_input = torch.cat([noise_vectors, one_hot_labels], dim=1)
        projected_features = self.projection_layer(combined_input)
        feature_map = projected_features.view(
            -1,
            GENERATOR_INITIAL_CHANNELS,
            GENERATOR_INITIAL_SPATIAL_SIZE,
            GENERATOR_INITIAL_SPATIAL_SIZE,
        )
        return self.upsampling_layers(feature_map)


class Discriminator(nn.Module):
    """Judges whether an image, given its claimed label, is real or fake.

    Educational note: the class label is supplied to the discriminator as
    a set of additional constant-valued channels, one per class,
    concatenated onto the image channel. This lets a single convolutional
    network reason jointly about "does this look real" and "does it look
    like the claimed category" without needing a separate network per
    class.
    """

    def __init__(self) -> None:
        """Build the convolutional layers of the discriminator."""
        super().__init__()
        input_channels = 1 + NUMBER_OF_CLASSES

        self.convolutional_layers = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(input_channels, DISCRIMINATOR_BASE_CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 14x14 -> 7x7
            nn.Conv2d(
                DISCRIMINATOR_BASE_CHANNELS,
                DISCRIMINATOR_BASE_CHANNELS * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(DISCRIMINATOR_BASE_CHANNELS * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        flattened_feature_size = (DISCRIMINATOR_BASE_CHANNELS * 2) * 7 * 7
        # A single raw logit is produced (no sigmoid), since training uses
        # BCEWithLogitsLoss for improved numerical stability.
        self.classification_layer = nn.Linear(flattened_feature_size, 1)

    def forward(self, images: Tensor, one_hot_labels: Tensor) -> Tensor:
        """Judge a batch of images.

        Parameters
        ----------
        images:
            A tensor of shape ``(batch_size, 1, IMAGE_HEIGHT,
            IMAGE_WIDTH)`` with pixel values in ``[-1, 1]``.
        one_hot_labels:
            A tensor of shape ``(batch_size, NUMBER_OF_CLASSES)``.

        Returns
        -------
        Tensor
            A tensor of shape ``(batch_size, 1)`` containing a raw
            "realness" logit (higher means more likely to be judged
            real).
        """
        batch_size = images.size(0)
        label_channels = one_hot_labels.view(batch_size, NUMBER_OF_CLASSES, 1, 1)
        label_channels = label_channels.expand(-1, -1, IMAGE_HEIGHT, IMAGE_WIDTH)

        combined_input = torch.cat([images, label_channels], dim=1)
        features = self.convolutional_layers(combined_input)
        flattened_features = torch.flatten(features, start_dim=1)
        return self.classification_layer(flattened_features)


# -------------------------------------------------------------------------------------------------
# 3. Data Loading
# -------------------------------------------------------------------------------------------------
def load_training_dataloader() -> DataLoader:
    """Download (if necessary) and load the Fashion-MNIST training set.

    Returns
    -------
    DataLoader
        A ``DataLoader`` yielding batches of ``(images, labels)``, with
        image pixel values normalised to ``[-1, 1]`` to match the
        generator's ``Tanh`` output range.

    Raises
    ------
    RuntimeError
        If the dataset could not be downloaded or loaded, for example due
        to a lack of internet access on first run.
    """
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )

    try:
        training_dataset = torchvision.datasets.FashionMNIST(
            root=str(DATA_DIRECTORY),
            train=True,
            download=True,
            transform=image_transform,
        )
    except Exception as error:  # noqa: BLE001 - re-raised with a clearer message below.
        raise RuntimeError(
            "Failed to download or load the Fashion-MNIST dataset. Check "
            "your internet connection and try again."
        ) from error

    return DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# -------------------------------------------------------------------------------------------------
# 4. Training
# -------------------------------------------------------------------------------------------------
def train_model(device: torch.device) -> Generator:
    """Train a fresh conditional DCGAN on Fashion-MNIST.

    Parameters
    ----------
    device:
        The device to train on.

    Returns
    -------
    Generator
        The trained generator, in evaluation mode. The discriminator is
        discarded after training, since it plays no role in generating
        new images -- it exists purely to provide a training signal.
    """
    training_dataloader = load_training_dataloader()

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    generator_optimiser = torch.optim.Adam(
        generator.parameters(), lr=LEARNING_RATE, betas=(ADAM_BETA_1, 0.999)
    )
    discriminator_optimiser = torch.optim.Adam(
        discriminator.parameters(), lr=LEARNING_RATE, betas=(ADAM_BETA_1, 0.999)
    )
    adversarial_loss_function = nn.BCEWithLogitsLoss()

    for epoch_index in range(1, NUMBER_OF_TRAINING_EPOCHS + 1):
        running_discriminator_loss = 0.0
        running_generator_loss = 0.0
        number_of_batches = 0

        for real_images, real_labels in training_dataloader:
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)
            real_one_hot_labels = functional.one_hot(real_labels, NUMBER_OF_CLASSES).float().to(device)

            # ----------------------------------------------------------
            # Step 1: update the discriminator.
            #
            # It is shown a batch of real images (which it should learn
            # to call "real") and a batch of freshly generated fake
            # images (which it should learn to call "fake").
            # ----------------------------------------------------------
            discriminator_optimiser.zero_grad()

            real_predictions = discriminator(real_images, real_one_hot_labels)
            real_targets = torch.full_like(real_predictions, REAL_LABEL_SMOOTHING_VALUE)
            real_loss = adversarial_loss_function(real_predictions, real_targets)

            noise_vectors = torch.randn(current_batch_size, LATENT_DIMENSIONALITY, device=device)
            random_labels = torch.randint(0, NUMBER_OF_CLASSES, (current_batch_size,), device=device)
            random_one_hot_labels = functional.one_hot(random_labels, NUMBER_OF_CLASSES).float()

            # .detach() prevents gradients from flowing into the generator
            # during the discriminator's update step.
            fake_images = generator(noise_vectors, random_one_hot_labels).detach()
            fake_predictions = discriminator(fake_images, random_one_hot_labels)
            fake_targets = torch.zeros_like(fake_predictions)
            fake_loss = adversarial_loss_function(fake_predictions, fake_targets)

            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            discriminator_optimiser.step()

            # ----------------------------------------------------------
            # Step 2: update the generator.
            #
            # It generates a fresh batch of fake images and is rewarded
            # when the discriminator is fooled into judging them "real".
            # ----------------------------------------------------------
            generator_optimiser.zero_grad()

            noise_vectors = torch.randn(current_batch_size, LATENT_DIMENSIONALITY, device=device)
            random_labels = torch.randint(0, NUMBER_OF_CLASSES, (current_batch_size,), device=device)
            random_one_hot_labels = functional.one_hot(random_labels, NUMBER_OF_CLASSES).float()

            fake_images = generator(noise_vectors, random_one_hot_labels)
            fooled_predictions = discriminator(fake_images, random_one_hot_labels)
            generator_targets = torch.ones_like(fooled_predictions)
            generator_loss = adversarial_loss_function(fooled_predictions, generator_targets)

            generator_loss.backward()
            generator_optimiser.step()

            running_discriminator_loss += discriminator_loss.item()
            running_generator_loss += generator_loss.item()
            number_of_batches += 1

        average_discriminator_loss = running_discriminator_loss / number_of_batches
        average_generator_loss = running_generator_loss / number_of_batches
        print(
            f"Epoch {epoch_index}/{NUMBER_OF_TRAINING_EPOCHS} - "
            f"discriminator loss: {average_discriminator_loss:.4f}, "
            f"generator loss: {average_generator_loss:.4f}"
        )

    generator.eval()
    return generator


def save_checkpoint(generator: Generator, checkpoint_path: Path) -> None:
    """Save the trained generator's weights to disk.

    Parameters
    ----------
    generator:
        The trained generator. The discriminator is intentionally not
        saved, since it is not required for generating new images.
    checkpoint_path:
        Where to save the ``.pth`` checkpoint.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"generator_state_dict": generator.state_dict()}, checkpoint_path)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Generator | None:
    """Attempt to load a previously trained generator from disk.

    Parameters
    ----------
    checkpoint_path:
        The path to a ``.pth`` checkpoint.
    device:
        The device to load the model onto.

    Returns
    -------
    Generator | None
        The loaded generator in evaluation mode, or ``None`` if no valid
        checkpoint could be loaded.
    """
    if not checkpoint_path.is_file():
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dictionary = (
            checkpoint.get("generator_state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        generator = Generator()
        generator.load_state_dict(state_dictionary)
    except Exception:  # noqa: BLE001 - any failure here just means "train fresh instead".
        return None

    generator.to(device)
    generator.eval()
    return generator


def obtain_trained_generator(device: torch.device) -> Generator:
    """Load a cached, trained generator if available, otherwise train a new one.

    Parameters
    ----------
    device:
        The device to train or load the model on.

    Returns
    -------
    Generator
        A trained generator, ready for image synthesis.
    """
    cached_generator = load_checkpoint(MODEL_CHECKPOINT_PATH, device)
    if cached_generator is not None:
        print(f"Loaded cached generator from '{MODEL_CHECKPOINT_PATH}'.")
        return cached_generator

    print(
        "No cached generator found. Training a new conditional DCGAN "
        "(this typically takes several minutes on CPU)..."
    )
    trained_generator = train_model(device)
    save_checkpoint(trained_generator, MODEL_CHECKPOINT_PATH)
    print(f"Saved trained generator to '{MODEL_CHECKPOINT_PATH}'.")
    return trained_generator


@torch.no_grad()
def generate_clothing_images(generator: Generator, device: torch.device) -> npt.NDArray[np.uint8]:
    """Generate several sample images for every clothing category.

    Parameters
    ----------
    generator:
        The trained generator.
    device:
        The device to generate on.

    Returns
    -------
    numpy.ndarray
        An array of shape ``(NUMBER_OF_CLASSES, SAMPLES_PER_CLASS,
        IMAGE_HEIGHT, IMAGE_WIDTH)`` containing generated images with
        pixel values in ``[0, 255]`` as ``uint8``.
    """
    all_generated_images = np.zeros(
        (NUMBER_OF_CLASSES, SAMPLES_PER_CLASS, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8
    )

    for class_index in range(NUMBER_OF_CLASSES):
        noise_vectors = torch.randn(SAMPLES_PER_CLASS, LATENT_DIMENSIONALITY, device=device)
        labels_for_this_class = torch.full(
            (SAMPLES_PER_CLASS,), fill_value=class_index, dtype=torch.long, device=device
        )
        one_hot_labels = functional.one_hot(labels_for_this_class, NUMBER_OF_CLASSES).float()

        generated_images = generator(noise_vectors, one_hot_labels)
        generated_images = generated_images.squeeze(1).cpu().numpy()

        # Undo the Tanh output's [-1, 1] range to recover [0, 255].
        generated_images_uint8 = (((generated_images + 1.0) / 2.0) * 255.0).clip(0, 255).astype(
            np.uint8
        )
        all_generated_images[class_index] = generated_images_uint8

    return all_generated_images


def build_labelled_grid(generated_images: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Arrange generated images into a single labelled BGR grid image.

    Each row corresponds to one clothing category (with its name printed
    on the left) and each column is a different randomly generated sample
    for that category.

    Parameters
    ----------
    generated_images:
        An array of shape ``(NUMBER_OF_CLASSES, SAMPLES_PER_CLASS,
        IMAGE_HEIGHT, IMAGE_WIDTH)``, as produced by
        :func:`generate_clothing_images`.

    Returns
    -------
    numpy.ndarray
        A BGR image of shape ``(NUMBER_OF_CLASSES * DISPLAY_TILE_SIZE,
        LABEL_MARGIN_WIDTH + SAMPLES_PER_CLASS * DISPLAY_TILE_SIZE, 3)``.
    """
    grid_height = NUMBER_OF_CLASSES * DISPLAY_TILE_SIZE
    grid_width = LABEL_MARGIN_WIDTH + SAMPLES_PER_CLASS * DISPLAY_TILE_SIZE
    grid_image = np.full((grid_height, grid_width, 3), fill_value=255, dtype=np.uint8)

    for class_index in range(NUMBER_OF_CLASSES):
        row_top = class_index * DISPLAY_TILE_SIZE

        text_position = (10, row_top + DISPLAY_TILE_SIZE // 2 + 5)
        cv2.putText(
            grid_image,
            CLASS_NAMES[class_index],
            text_position,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        for sample_index in range(SAMPLES_PER_CLASS):
            small_image = generated_images[class_index, sample_index]
            upscaled_image = cv2.resize(
                small_image,
                (DISPLAY_TILE_SIZE, DISPLAY_TILE_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )
            upscaled_image_bgr = cv2.cvtColor(upscaled_image, cv2.COLOR_GRAY2BGR)

            column_left = LABEL_MARGIN_WIDTH + sample_index * DISPLAY_TILE_SIZE
            grid_image[
                row_top : row_top + DISPLAY_TILE_SIZE,
                column_left : column_left + DISPLAY_TILE_SIZE,
            ] = upscaled_image_bgr

    return grid_image


def save_grid_image(grid_image: npt.NDArray[np.uint8], output_path: Path) -> None:
    """Save the generated image grid to disk.

    Parameters
    ----------
    grid_image:
        The BGR grid image to save.
    output_path:
        The destination file path.

    Raises
    ------
    RuntimeError
        If OpenCV fails to write the image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_succeeded = cv2.imwrite(str(output_path), grid_image)
    if not write_succeeded:
        raise RuntimeError(f"OpenCV failed to save the output image to '{output_path}'.")


def select_computation_device() -> torch.device:
    """Choose the best available device for training and generation.

    Returns
    -------
    torch.device
        ``cuda`` if a compatible GPU is available, otherwise ``cpu``.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main() -> None:
    """Run the end-to-end training-and-generation demonstration."""
    try:
        device = select_computation_device()
        generator = obtain_trained_generator(device)

        generated_images = generate_clothing_images(generator, device)
        grid_image = build_labelled_grid(generated_images)
        save_grid_image(grid_image, GENERATED_GRID_PATH)

    except RuntimeError as error:
        print(f"Demonstration failed: {error}")
        return

    print(f"Generation complete. Saved result to '{GENERATED_GRID_PATH}'.")

if __name__ == "__main__":
    main()