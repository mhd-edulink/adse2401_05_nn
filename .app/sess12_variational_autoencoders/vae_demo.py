"""
=======================================================================================
Python script to demonstrate a single-image Variational Autoencoder
=======================================================================================

This program demonstrates image compression using a variational autoencoder. It's
intentionally simplified for clarity rather than performance.

SCRIPT OVERVIEW:
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Running this script will
1. Load the image from 'files/original.jpg'
2. Resize and convert it to a tensor.
3. Build a small VAE (encoder and decoder) from scratch in PyTorch.
4. Intentionally overfit the VAE to this (original.jpg) image
   in the training section below
5. Use the trained VAE to "compress" (encoding) and "decompress" (decoding) the image.
6. Save the reconstruction to 'files/reconstruction.png'
7. Display the reconsturcted and original image side-by-side
8. Print conceptual compression statistics.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Requirements:
    !pip install matplotlib numpy pillow torch torchvision tqdm
"""

# -----------------------------------------------------------------------
# 0. Import required modules
# -----------------------------------------------------------------------
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as functional

from pathlib import Path
from PIL import Image
from torch import Tensor
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm
from typing import Tuple

import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------
# 1. Constants
# -----------------------------------------------------------------------
# Directory containing the input image and where the outputs will be written
FILES_DIRECTORY: Path = Path("../files")

# Path to the input image
ORIGINAL_IMAGE_PATH: Path = FILES_DIRECTORY / "original.jpg"

# Path where the reconstructed image will be saved
RECONSTRUCTED_IMAGE_PATH: Path = FILES_DIRECTORY / "reconstruction.png"

# Resize the image to a fixed square resolutions.
# A fixed size keeps the network architecture simple,
# since fully connected layers require a known, constant input value
IMAGE_SIZE: int = 128

# Number of colour channels. We work with RGB, hence 3
NUMBER_OF_CHANNELS: int = 3

# Dimensionality of the VAE's latent space (the "compressed" representation of the image) 64 is a good value
LATENT_DIMENSIONS: int = 64

# Sized of the hidden fully connected layers in the encoder and decoder
HIDDEN_LAYER_ONE_SIZE: int = 1024
HIDDEN_LAYER_TWO_SIZE: int = 256

# Training hyperparameters
NUMBER_OF_EPOCHS: int = 300
LEARNING_RATE: float = 1e-3

# Weight applied to the KL_DIVERGENCE term in the loss function
KL_DIVERGENCE_WEIGHT: float = 1.0  # Set to 1.0 for a "vanilla" VAE

# Preprocessing device on which all tensors and models will live. Prefer PU (CUDA) if available or CPU if not
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed random seed for reproducibility from run to run
RANDOME_SEED: int = 42


# -----------------------------------------------------------------------
# 2. Utility Function
# -----------------------------------------------------------------------
def set_random_seed(seed: int) -> None:
    """
    Set the random seed for Numpy and PyTorch.
    :param seed:
        The integer seed to apply to Numpy's and Pytorch's random numbers
    :return:
        None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_and_preprocess_image(image_path: Path, image_size: int) -> Tensor:
    """Load an image from disk and convert it into a normalised tensor.

    This function demonstrates the standard "load, resize, convert to
    tensor" pipeline that precedes almost all computer vision deep
    learning models.

    Parameters
    ----------
    image_path:
        The path to the JPEG (or other Pillow-readable) image file.
    image_size:
        The width and height, in pixels, to resize the (square) image to.

    Returns
    -------
    Tensor
        A tensor of shape ``(1, channels, image_size, image_size)`` with
        pixel values scaled to the range ``[0, 1]``. The leading dimension
        of size 1 is the "batch" dimension that PyTorch models expect,
        even though we only have a single image.

    Raises
    ------
    FileNotFoundError
        If no image exists at ``image_path``.
    """

    if not image_path.exists():
        raise FileNotFoundError(f"Could not find input image at '{image_path}'"
                                f" Please place a JPEG named 'original.jpg' at {ORIGINAL_IMAGE_PATH}"
                                f" before running this script.")

    # Pillow loads the raw image file; we force conversion to RGB in case the source image
    # is greyscale or has an alpha channel
    pil_image = Image.open(image_path).convert("RGB")

    # torchvision.transforms.Compose lets us chain several preprocessing
    # steps together. Resize changes the spatial resolution; ToTensor
    # converts the Pillow image into a PyTorch tensor and automatically
    # rescales pixel values from [0, 255] integers to [0.0, 1.0] floats,
    # which is the range our network's sigmoid output will also use.
    # Create the preprocessing pipeline
    preprocessing_pipeline = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    # Apply the preprocessing pipeline
    image_tensor = preprocessing_pipeline(pil_image)

    # Add the batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor.to(DEVICE)


def tensor_to_numpy_image(image_tensor: Tensor) -> np.ndarray:
    """Convert a normalised image tensor back into a displayable array.

    Parameters
    ----------
    image_tensor:
        A tensor of shape ``(1, channels, height, width)`` with values in
        the range ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        An array of shape ``(height, width, channels)`` with values in
        ``[0, 1]``, suitable for display with ``matplotlib``.
    """
    # Detach from the autograd graph, move to the CPU, and drop the batch
    # dimension before rearranging the axes for matplotlib, which expects
    # channels last (H, W, C) rather than PyTorch's channels-first (C, H, W).
    image = image_tensor.detach().cpu().squeeze(0)
    image = image.permute(1, 2, 0).numpy()
    image = np.clip(image, 0.0, 1.0)
    return image


def format_output(content: str) -> None:
    """Print a neatly formatted heading describing the current stage.

    Used throughout the program to give students a clear, sequential
    narrative of what the code is doing at any given moment - this is
    intended for projection during a lecture or lab session.

    Parameters
    ----------
    content:
        A short human-readable description of the stage about to begin.
    """
    print(f"\n" + "=" * 70)
    print(content)
    print(f"\n" + "=" * 70)


# -----------------------------------------------------------------------------
# 3. Encoder class
# -----------------------------------------------------------------------------
class Encoder(nn.Module):
    """The encoder half of the Variational Autoencoder.

    The encoder's job is to compress an input image into a low-dimensional
    latent representation. Unlike a standard (deterministic) autoencoder,
    a VAE's encoder does not output a single latent vector directly.
    Instead, it outputs the *parameters of a probability distribution*
    (specifically, a diagonal Gaussian) over the latent space: a mean
    vector ``mu`` and a log-variance vector ``log_var``.

    Why output a distribution rather than a single point?
    -------------------------------------------------------
    If we only learned a single latent vector per image (as a plain
    autoencoder does), the latent space could become an arbitrary,
    disconnected scattering of points with "gaps" that do not correspond
    to any realistic image. By instead forcing each image to map to a
    small *region* (a Gaussian "cloud") of latent space, and by
    encouraging nearby regions to overlap (via the KL divergence term
    explained later), the latent space becomes smooth and continuous.
    This means that points we have never explicitly trained on - for
    example, points interpolated between two training images - still
    decode into plausible-looking images.

    We predict ``log_var`` rather than ``var`` directly for numerical
    convenience: the network's raw output can be any real number,
    whereas a variance must be positive. Exponentiating ``log_var``
    guarantees a positive variance without needing extra constraints
    on the network's weights.
    """

    def __init__(self, image_size: int, number_of_channels: int, latent_dimension: int) -> None:
        """Construct the encoder network.

        Parameters
        ----------
        image_size:
            The height and width (in pixels) of the square input images.
        number_of_channels:
            The number of colour channels in the input images (3 for RGB).
        latent_dimension:
            The size of the latent vector the encoder should produce.
        """
        super().__init__()

        # We flatten the image into a sngle long vector and use fully connected
        # ("dense") layers throughout. Convolutionsal layers are usually preferred
        # for images, but fully connected network is safer
        self.input_dimension = number_of_channels * image_size * image_size

        # A stack of two hidden layers gradually reduces the dimensionality
        # from the raw pixel count down towards the latent dimension.
        # ReLU is used as our non-linearity - a standard, simple choice.
        self.shared_hidden_layers = nn.Sequential(
            nn.Linear(self.input_dimension, HIDDEN_LAYER_ONE_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_ONE_SIZE, HIDDEN_LAYER_TWO_SIZE),
            nn.ReLU(),
        )

        # Two separate output "heads" branch off from the shared hidden
        # representation: one predicts the mean (mu) of the latent
        # distribution, and the other predicts the log-variance
        # (log_var). Both have the same shape: (latent_dimension,).
        self.mean_head = nn.Linear(HIDDEN_LAYER_TWO_SIZE, latent_dimension)
        self.log_variance_head = nn.Linear(HIDDEN_LAYER_TWO_SIZE, latent_dimension)

    def forward(self, image_batch: Tensor) -> Tuple[Tensor, Tensor]:
        """Run a forward pass through the encoder.

        Parameters
        ----------
        image_batch:
            A batch of images with shape
            ``(batch_size, channels, height, width)``.

        Returns
        -------
        Tuple[Tensor, Tensor]
            A tuple ``(mu, log_var)``, each of shape
            ``(batch_size, latent_dimension)``, describing the mean and
            log-variance of the approximate posterior distribution
            ``q(z | x)`` for every image in the batch.
        """
        batch_size = image_batch.shape[0]

        # Flatten each image in the batch from (C, H, W) into a single
        # long vector, ready for the fully connected layers.
        flattened_images = image_batch.view(batch_size, -1)

        hidden_representation = self.shared_hidden_layers(flattened_images)

        mean = self.mean_head(hidden_representation)
        log_variance = self.log_variance_head(hidden_representation)

        return mean, log_variance


# -----------------------------------------------------------------------------
# 4. Decoder class
# -----------------------------------------------------------------------------
class Decoder(nn.Module):
    """The decoder half of the Variational Autoencoder.

    The decoder takes a latent vector ``z`` (sampled from the
    distribution produced by the encoder) and reconstructs an image from
    it. Architecturally, it mirrors the encoder: the same layer sizes are
    used, but in reverse, gradually expanding the compact latent vector
    back up to the full number of pixels.
    """

    def __init__(
            self,
            image_size: int,
            number_of_channels: int,
            latent_dimension: int,
    ) -> None:
        """Construct the decoder network.

        Parameters
        ----------
        image_size:
            The height and width (in pixels) of the square output images.
        number_of_channels:
            The number of colour channels to reconstruct (3 for RGB).
        latent_dimension:
            The size of the latent vector this decoder accepts as input.
        """
        super().__init__()

        self.image_size = image_size
        self.number_of_channels = number_of_channels
        self.output_dimension = number_of_channels * image_size * image_size

        # The decoder is the mirror image of the encoder: it expands the latent
        # vector backup through the same hdiden layer sizes, in reverse order,
        # before producing a full-size image vector

        self.network = nn.Sequential(
            nn.Linear(latent_dimension, HIDDEN_LAYER_TWO_SIZE),
            nn.ReLU(),

            nn.Linear(HIDDEN_LAYER_TWO_SIZE, HIDDEN_LAYER_ONE_SIZE),
            nn.ReLU(),

            nn.Linear(HIDDEN_LAYER_ONE_SIZE, self.output_dimension),
            nn.Sigmoid(),
        )

    def forward(self, latent_vector_batch: Tensor) -> Tensor:
        """Run a forward pass through the decoder.

        Parameters
        ----------
        latent_vector_batch:
            A batch of latent vectors with shape
            ``(batch_size, latent_dimension)``.

        Returns
        -------
        Tensor
            A batch of reconstructed images with shape
            ``(batch_size, channels, height, width)`` and pixel values in
            the range ``[0, 1]``.
        """
        batch_size = latent_vector_batch.shape[0]
        flat_reconstruction = self.network(latent_vector_batch)

        # Reshape the flat vector of pixels back into a proper image
        # tensor: (batch_size, channels, height, width)
        reconstructed_images = flat_reconstruction.view(
            batch_size,
            self.number_of_channels,
            self.image_size,
            self.image_size
        )
        return reconstructed_images


# -----------------------------------------------------------------------------
# 5. Variational Autoencoder
# -----------------------------------------------------------------------------
class VariationalAutoencoder(nn.Module):
    """
    A complete Variational Autoencoder combining an Encoder and Decoder.

    This class ties together the encoder, the reparameterisation trick,
    and the decoder into a single, end-to-end trainable model.
    Construct the VAE from an encoder and a decoder.

    Parameters
    ----------
    image_size:
        The height and width (in pixels) of the square images this
        VAE will operate on.
    number_of_channels:
        The number of colour channels in the images (3 for RGB).
    latent_dimension:
        The size of the compressed latent representation.
    """

    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        number_of_channels: int = NUMBER_OF_CHANNELS,
        latent_dimension: int = LATENT_DIMENSIONS,
    ) -> None:
        """Construct the VAE from an encoder and a decoder...."""
        super().__init__()

        self.encoder = Encoder(image_size, number_of_channels, latent_dimension)
        self.decoder = Decoder(image_size, number_of_channels, latent_dimension)
        self.latent_dimension = latent_dimension

    def decode(self, latent_vector: Tensor) -> Tensor:
        """
        Decode a latent vector back into an image.
        """
        return self.decoder(latent_vector)

    def forward(self, image_batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Run a full forward pass: encode, sample, then decode.

        Parameters
        ----------
        image_batch:
            A batch of input images, shape
            ``(batch_size, channels, height, width)``.

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor]
            A tuple ``(reconstruction, mean, log_variance)`` where
            ``reconstruction`` is the decoder's output image, and
            ``mean``/``log_variance`` describe the latent distribution
            produced by the encoder (needed later to compute the KL
            divergence term of the loss).
        """
        mean, log_variance = self.encoder(image_batch)
        latent_vector = self.reparameterise(mean, log_variance)
        reconstruction = self.decoder(latent_vector)
        return reconstruction, mean, log_variance

    def encode_to_latent_vector(self, image_batch: Tensor) -> Tensor:

        mean, _ = self.encoder(image_batch)
        return mean

    @staticmethod
    def reparameterise(mean: Tensor, log_variance: Tensor) -> Tensor:
        """Sample a latent vector using the reparameterisation trick...."""
        # Converting log-variance back to a standard deviation.
        # We work with log-variance for numerical stability (see the
        # Encoder docstring), and convert to a standard deviation here
        # purely for the sampling arithmetic itself.
        standard_deviation = torch.exp(0.5 * log_variance)

        # Epsilon is random "noise" drawn from the standard normal distribution; it
        # has no learnable parameters, so its safe to sample directlty without backpropagation.
        epsilon = torch.randn_like(standard_deviation)

        # The reparameterisation trick
        sampled_latent_vector = mean + standard_deviation * epsilon

        return sampled_latent_vector


# -----------------------------------------------------------------------------
# 6. Loss Function
# -----------------------------------------------------------------------------
def compute_vae_loss(reconstructed_images: Tensor, original_images: Tensor, mean: Tensor,
                     log_variance: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    # Reconstruction loss: summed squared pixel-wise error. We use "sum"
    # reduction (rather than "mean") so that term is on a comparable scale to the KL divergence
    # which is also a sum over latent dimensions. This is standard practice in VAE implementations
    reconstruction_loss = functional.mse_loss(reconstructed_images, original_images, reduction="sum")

    # KL divergence between the encoder's distribution N(Mean, variance)
    # and the standard normal prior N(0, 1). For two diagonal Gaussians,
    # this has a convinient closed-form expression:
    #
    #   KL = -0.5 * sum(1 + log_variance - mean^2 - exp(log_variance))
    #
    # Intuitevely: this term is zeor when mean = 0 and
    # variance = 1 (i.e when the encoder's ouput matches the prior),
    # and grows whenever the encoder pushes the mean away from zero or
    # the variance away from one
    kl_divergence = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())

    total_loss = reconstruction_loss + KL_DIVERGENCE_WEIGHT * kl_divergence
    return total_loss, reconstruction_loss, kl_divergence


# -----------------------------------------------------------------------------
# 7. Training Function
# -----------------------------------------------------------------------------
def train_vae_on_single_image(
        model: VariationalAutoEncoder,
        image_tensor: Tensor,
        number_of_epochs: int,
        learning_rate: float,
) -> float:
    optimiser = Adam(model.parameters(), lr=learning_rate)
    model.train()
    final_reconstruction_loss = 0.0

    # tqdm wraps our epoch range to give a live, visual progress bar in the terminal
    progress_bar = tqdm(range(number_of_epochs), desc="Training VAE", unit="epoch")

    for epoch in progress_bar:
        optimiser.zero_grad()
        reconstruction, mean, log_variance = model(image_tensor)
        total_loss, reconstruction_loss, kl_divergence = compute_vae_loss(
            reconstruction, image_tensor, mean, log_variance
        )

        # Backpropogation: compute gradients of the total loss with
        # respect to every trainable parameter in the encoder and
        # decoder, thanks to the differentiable reparameterisation
        total_loss.backward()
        optimiser.step()

        final_reconstruction_loss += reconstruction_loss.item()

        # Update the progress bar with concise, live-updateing numbers
        # rather than prininting a new line every epoch which would
        # quickly flood the console
        progress_bar.set_postfix({
            "total_loss": f"{total_loss.item():.2f}",
            "reconstruction": f"{reconstruction_loss.item():.2f}",
            "kl_divergence": f"{kl_divergence.item():.2f}",
        })

        return final_reconstruction_loss


# -----------------------------------------------------------------------------
# 8. Visualisation
# -----------------------------------------------------------------------------
def display_and_save_comparison(
        original_image_tensor: Tensor,
        reconstructed_image_tensor: Tensor,
        output_path: Path
) -> None:
    original_image_array = tensor_to_numpy_image(original_image_tensor)
    reconstructed_image_array = tensor_to_numpy_image(reconstructed_image_tensor)

    # Save the reconstruction to disk first, independently of the matplot display,
    # using Pillow so that the save file does not depend on matplotlib's figure rendering settings.
    reconstructed_uint8 = (reconstructed_image_array * 255).astype(np.uint8)
    Image.fromarray(reconstructed_uint8).save(output_path)

    # Now build a simple side-by-side comparision figure
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    axes[0].imshow(original_image_array)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(reconstructed_image_array)
    axes[1].set_title("Reconstructed Image (VAE)")
    axes[1].axis("off")

    figure.suptitle("VAE Image Compression Demonstration")
    figure.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 9. Main Execution Function
# -----------------------------------------------------------------------------
def main() -> None:
    set_random_seed(RANDOME_SEED)

    # -----------------------------------------------------------------------------
    # Stage I. Load image
    # -----------------------------------------------------------------------------
    format_output("Stage I: Loading the original image")
    print(f"Reading image from: {ORIGINAL_IMAGE_PATH}")

    try:
        original_image_tensor = load_and_preprocess_image(ORIGINAL_IMAGE_PATH, IMAGE_SIZE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Image loaded and resized to {IMAGE_SIZE} x {IMAGE_SIZE}."
          f"\nTensor Shape: {tuple(original_image_tensor.shape)}"
          f"\nRunning on device: {DEVICE}")

    # -----------------------------------------------------------------------------
    # Stage II. Building the encoder and decoder.
    # -----------------------------------------------------------------------------
    format_output("Stage II: Building the encoder and decoder networks")
    print(
        f"The encoder copresses a"
        f"\n{IMAGE_SIZE} x {IMAGE_SIZE} x {NUMBER_OF_CHANNELS} image down to"
        f"a {LATENT_DIMENSIONS}-dimensional latent distribution (mean and log variance)"
    )

    vae_model = VariationalAutoencoder(
        image_size=IMAGE_SIZE,
        number_of_channels=NUMBER_OF_CHANNELS,
        latent_dimension=LATENT_DIMENSIONS
    ).to(DEVICE)

    number_of_parameters = sum(
        paramaeter.numel() for paramaeter in vae_model.parameters()
    )
    print(f"Total trainable parameters: {number_of_parameters:,}")

    # -----------------------------------------------------------------------------
    # Stage III. Training (deliberately overfitting to one image)
    # -----------------------------------------------------------------------------
    format_output("Stage III: Training the VAE on the single output image")
    print(f"By Training with only one training image, the network is intentionally "
          f"being overfitten. This is not how VAEs are used in practice, but it lets us watch, epoch "
          f"by epoch, how the network learns to encode and reconstruct one specific image"
          f"through a narrow latent bottleneck. ")
    final_reconstruction_loss = train_vae_on_single_image(vae_model, original_image_tensor,
                                                          number_of_epochs=NUMBER_OF_EPOCHS,
                                                          learning_rate=LEARNING_RATE)

    # -----------------------------------------------------------------------------
    # Stage IV. Compressing the image into its latent representation.
    # -----------------------------------------------------------------------------
    format_output("Stage IV: Compressing the image into its latent vector")

    vae_model.eval()
    with torch.no_grad():
        latent_vector = vae_model.encode_to_latent_vector(original_image_tensor)

    print(f"Latent vector shape: {tuple(latent_vector.shape)}")
    print(
        "This latent vector is the VAE's compressed representation of "
        "the original image - single point summarising the picture in just "
        f"{LATENT_DIMENSIONS} floating-point numbers. "
    )

    # -----------------------------------------------------------------------------
    # Stage V. Reconstructing the image from the latent vector
    # -----------------------------------------------------------------------------
    format_output("Stage V: Reconstructing the image from the latent vector.")

    with torch.no_grad():
        reconstructed_image_tensor = vae_model.decode(latent_vector)
    print(f"Reconstruction complete.")
    print(
        "Note: VAE reconstructions typically look slightly "
        "blurred compared with the original. This happens because the "
        "reconstruction loss (mean squared error) encourages the "
        "decoder to output the *average* of plausible pixel values "
        "wherever it is uncertain, rather than committing to sharp, "
        "high-frequency detail. Generative Adversarial Networks (GANs) "
        "and diffusion models use different training objectives (an "
        "adversarial discriminator, or iterative denoising) that tend "
        "to encourage sharper, more detailed outputs."
    )

    # -----------------------------------------------------------------------------
    # Stage VI Saving the output.
    # -----------------------------------------------------------------------------
    format_output("Stage VI: Saving and displaying the reconstructed image.")
    print(f"Saving reconstructed image to {RECONSTRUCTED_IMAGE_PATH}")

    display_and_save_comparison(
        original_image_tensor, reconstructed_image_tensor, RECONSTRUCTED_IMAGE_PATH
    )

    # -----------------------------------------------------------------------------
    # Stage VII. Compression Statistics
    # -----------------------------------------------------------------------------
    format_output("Stage VII: Compression Statistics.")
    number_of_pixels = IMAGE_SIZE * IMAGE_SIZE * NUMBER_OF_CHANNELS
    compression_ratio = number_of_pixels / LATENT_DIMENSIONS

    print(f"Original image dimensions: {IMAGE_SIZE} x {IMAGE_SIZE}"
          f" x {NUMBER_OF_CHANNELS} channels")
    print(f"Total number of pixels: {number_of_pixels}")
    print(f"Latent vector size: {LATENT_DIMENSIONS} values")
    print(f"Conceptual compression: {compression_ratio:.2f} : 1")
    print(f"Final reconstruction loss: {final_reconstruction_loss}")

# -----------------------------------------------------------------------------
# 10. Run the script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()