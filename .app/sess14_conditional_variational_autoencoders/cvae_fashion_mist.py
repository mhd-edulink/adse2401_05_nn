from __future__ import annotations

import warnings
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as functional
import torchvision

from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader


warnings.filterwarnings("ignore")


# ============================================================
# Constants
# ============================================================

IMAGE_HEIGHT: Final[int] = 28
IMAGE_WIDTH: Final[int] = 28
IMAGE_PIXEL_COUNT: Final[int] = IMAGE_HEIGHT * IMAGE_WIDTH

NUMBER_OF_CLASSES: Final[int] = 10

CLASS_NAMES = [
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

LATENT_DIMENSIONALITY = 20

ENCODER_HIDDEN_SIZE = 400
DECODER_HIDDEN_SIZE = 400

NUMBER_OF_TRAINING_EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

SAMPLES_PER_CLASS = 6
DISPLAY_TILE_SIZE = 96
LABEL_MARGIN_WIDTH = 120

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = SCRIPT_DIRECTORY / "data"
OUTPUT_DIRECTORY = SCRIPT_DIRECTORY / "output"

MODEL_CHECKPOINT_PATH = OUTPUT_DIRECTORY / "cvae_fashion_mnist.pth"
GENERATED_GRID_PATH = OUTPUT_DIRECTORY / "generated_clothing_grid.png"


# ============================================================
# Encoder
# ============================================================

class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.hidden = nn.Linear(
            IMAGE_PIXEL_COUNT + NUMBER_OF_CLASSES,
            ENCODER_HIDDEN_SIZE,
        )

        self.mean = nn.Linear(
            ENCODER_HIDDEN_SIZE,
            LATENT_DIMENSIONALITY,
        )

        self.log_variance = nn.Linear(
            ENCODER_HIDDEN_SIZE,
            LATENT_DIMENSIONALITY,
        )

    def forward(
        self,
        images: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:

        x = torch.cat((images, labels), dim=1)

        x = functional.relu(self.hidden(x))

        mean = self.mean(x)
        log_variance = self.log_variance(x)

        return mean, log_variance


# ============================================================
# Decoder
# ============================================================

class Decoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.hidden = nn.Linear(
            LATENT_DIMENSIONALITY + NUMBER_OF_CLASSES,
            DECODER_HIDDEN_SIZE,
        )

        self.output = nn.Linear(
            DECODER_HIDDEN_SIZE,
            IMAGE_PIXEL_COUNT,
        )

    def forward(
        self,
        latent_vectors: Tensor,
        labels: Tensor,
    ) -> Tensor:

        x = torch.cat((latent_vectors, labels), dim=1)

        x = functional.relu(self.hidden(x))

        x = torch.sigmoid(self.output(x))

        return x


# ============================================================
# Conditional Variational Autoencoder
# ============================================================

class ConditionalVariationalAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    @staticmethod
    def reparameterise(
        mean: Tensor,
        log_variance: Tensor,
    ) -> Tensor:

        standard_deviation = torch.exp(0.5 * log_variance)

        epsilon = torch.randn_like(standard_deviation)

        return mean + epsilon * standard_deviation

    def forward(
        self,
        images: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:

        mean, log_variance = self.encoder(images, labels)

        latent_vector = self.reparameterise(
            mean,
            log_variance,
        )

        reconstruction = self.decoder(
            latent_vector,
            labels,
        )

        return reconstruction, mean, log_variance

    @torch.no_grad()
    def generate(
        self,
        labels: Tensor,
        device: torch.device,
    ) -> Tensor:

        batch_size = labels.size(0)

        latent_vectors = torch.randn(
            batch_size,
            LATENT_DIMENSIONALITY,
            device=device,
        )

        return self.decoder(
            latent_vectors,
            labels,
        )

# ===============================================================================================
# 03. Data Loading
# ===============================================================================================

def load_training_dataloader() -> DataLoader:
    """
    Download (if necessary) and return the Fashion-MNIST training loader.
    """

    transform = torchvision.transforms.ToTensor()

    dataset = torchvision.datasets.FashionMNIST(
        root=DATA_DIRECTORY,
        train=True,
        download=True,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    return dataloader


# ===============================================================================================
# 04. Loss Function
# ===============================================================================================

def compute_vae_loss(
    reconstructed_images: Tensor,
    original_images: Tensor,
    latent_mean: Tensor,
    latent_log_variance: Tensor,
) -> Tensor:
    """
    Computes the standard VAE loss.

    Total Loss =
        Reconstruction Loss
        +
        KL Divergence
    """

    reconstruction_loss = functional.binary_cross_entropy(
        reconstructed_images,
        original_images,
        reduction="sum",
    )

    kl_divergence = -0.5 * torch.sum(
        1
        + latent_log_variance
        - latent_mean.pow(2)
        - latent_log_variance.exp()
    )

    return reconstruction_loss + kl_divergence


# ===============================================================================================
# 05. Checkpoint Utilities
# ===============================================================================================

def save_checkpoint(
    model: ConditionalVariationalAutoencoder,
    checkpoint_path: Path,
) -> None:

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> ConditionalVariationalAutoencoder | None:

    if not checkpoint_path.exists():
        return None

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        state_dictionary = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict)
            else checkpoint
        )

        model = ConditionalVariationalAutoencoder()

        model.load_state_dict(state_dictionary)

        model.to(device)

        model.eval()

        return model

    except Exception:

        return None


# ===============================================================================================
# 06. Device Selection
# ===============================================================================================

def select_computation_device() -> torch.device:

    if torch.cuda.is_available():

        print("Using GPU")

        return torch.device("cuda")

    print("Using CPU")

    return torch.device("cpu")

# ===============================================================================================
# 07. Training
# ===============================================================================================

def train_model(
    device: torch.device,
) -> ConditionalVariationalAutoencoder:

    dataloader = load_training_dataloader()

    model = ConditionalVariationalAutoencoder().to(device)

    optimiser = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    model.train()

    dataset_size = len(dataloader.dataset)

    for epoch in range(NUMBER_OF_TRAINING_EPOCHS):

        running_loss = 0.0

        for images, labels in dataloader:

            images = images.to(device)

            labels = labels.to(device)

            flattened_images = images.view(
                images.size(0),
                IMAGE_PIXEL_COUNT,
            )

            one_hot_labels = functional.one_hot(
                labels,
                NUMBER_OF_CLASSES,
            ).float()

            optimiser.zero_grad()

            reconstructed_images, latent_mean, latent_log_variance = model(
                flattened_images,
                one_hot_labels,
            )

            loss = compute_vae_loss(
                reconstructed_images,
                flattened_images,
                latent_mean,
                latent_log_variance,
            )

            loss.backward()

            optimiser.step()

            running_loss += loss.item()

        average_loss = running_loss / dataset_size

        print(
            f"Epoch "
            f"{epoch + 1:02d}"
            f"/{NUMBER_OF_TRAINING_EPOCHS}"
            f"   Loss: {average_loss:.4f}"
        )

    model.eval()

    return model


# ===============================================================================================
# 08. Cached Model Helper
# ===============================================================================================

def obtain_trained_model(
    device: torch.device,
) -> ConditionalVariationalAutoencoder:

    cached_model = load_checkpoint(
        MODEL_CHECKPOINT_PATH,
        device,
    )

    if cached_model is not None:

        print(
            f"Loaded cached model:\n"
            f"{MODEL_CHECKPOINT_PATH}"
        )

        return cached_model

    print(
        "No cached model found.\n"
        "Training a new CVAE..."
    )

    model = train_model(device)

    save_checkpoint(
        model,
        MODEL_CHECKPOINT_PATH,
    )

    print(
        f"Checkpoint saved to:\n"
        f"{MODEL_CHECKPOINT_PATH}"
    )

    return model


# ===============================================================================================
# 09. Generation
# ===============================================================================================

def generate_clothing_images(
    model: ConditionalVariationalAutoencoder,
    device: torch.device,
) -> npt.NDArray[np.uint8]:

    all_generated_images = np.zeros(
        (
            NUMBER_OF_CLASSES,
            SAMPLES_PER_CLASS,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        ),
        dtype=np.uint8,
    )

    for class_index in range(NUMBER_OF_CLASSES):

        labels = torch.full(
            (SAMPLES_PER_CLASS,),
            class_index,
            dtype=torch.long,
            device=device,
        )

        one_hot_labels = functional.one_hot(
            labels,
            NUMBER_OF_CLASSES,
        ).float()

        generated_images = model.generate(
            one_hot_labels,
            device,
        )

        generated_images = (
            generated_images.cpu().numpy()
            * 255
        ).clip(
            0,
            255,
        ).astype(
            np.uint8
        )

        generated_images = generated_images.reshape(
            SAMPLES_PER_CLASS,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        )

        all_generated_images[class_index] = generated_images

    return all_generated_images


# ===============================================================================================
# 10. Visualisation
# ===============================================================================================

def build_labelled_grid(
    generated_images: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:

    grid_height = NUMBER_OF_CLASSES * DISPLAY_TILE_SIZE

    grid_width = (
        LABEL_MARGIN_WIDTH
        + SAMPLES_PER_CLASS * DISPLAY_TILE_SIZE
    )

    grid = np.full(
        (
            grid_height,
            grid_width,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    for class_index in range(NUMBER_OF_CLASSES):

        row = class_index * DISPLAY_TILE_SIZE

        cv2.putText(
            grid,
            CLASS_NAMES[class_index],
            (10, row + DISPLAY_TILE_SIZE // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        for sample_index in range(SAMPLES_PER_CLASS):

            image = generated_images[
                class_index,
                sample_index,
            ]

            image = cv2.resize(
                image,
                (
                    DISPLAY_TILE_SIZE,
                    DISPLAY_TILE_SIZE,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR,
            )

            left = (
                LABEL_MARGIN_WIDTH
                + sample_index * DISPLAY_TILE_SIZE
            )

            grid[
                row:row + DISPLAY_TILE_SIZE,
                left:left + DISPLAY_TILE_SIZE,
            ] = image

    return grid


def save_grid_image(
    grid_image: npt.NDArray[np.uint8],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(output_path),
        grid_image,
    ):

        raise RuntimeError(
            "Failed to save image."
        )


# ===============================================================================================
# 11. Main
# ===============================================================================================

def main() -> None:

    try:

        device = select_computation_device()

        model = obtain_trained_model(device)

        generated_images = generate_clothing_images(
            model,
            device,
        )

        grid_image = build_labelled_grid(
            generated_images,
        )

        save_grid_image(
            grid_image,
            GENERATED_GRID_PATH,
        )

    except RuntimeError as error:

        print(error)

        return

    print()

    print("==========================================")

    print("Conditional VAE completed successfully.")

    print(f"Results saved to:")

    print(GENERATED_GRID_PATH)

    print("==========================================")


# ===============================================================================================
# 12. Script Entry Point
# ===============================================================================================

if __name__ == "__main__":
    main()