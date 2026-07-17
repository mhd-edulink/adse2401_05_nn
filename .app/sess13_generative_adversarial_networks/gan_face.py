"""
gan_faces.py

Educational DCGAN for CelebA face generation.
"""

import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

IMAGE_SIZE = 64
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
EPOCHS = 50
LATENT_VECTOR_SIZE = 100
BETA1 = 0.5
NUMBER_OF_WORKERS = 0
RANDOM_SEED = 42
CHANNELS = 3

ROOT = Path(__file__).resolve().parent
MALE_FOLDER = ROOT / "../files" / "male"
FEMALE_FOLDER = ROOT / "../files" / "female"
GENERATED_FOLDER = ROOT / "../files" / "generated"
MODEL_FOLDER = ROOT / "saved_models"

GENERATOR_MODEL_PATH = MODEL_FOLDER / "generator.pth"
DISCRIMINATOR_MODEL_PATH = MODEL_FOLDER / "discriminator.pth"

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def get_device():
    if torch.cuda.is_available():
        print("Running on NVIDIA GPU")
        return torch.device("cuda")
    print("Running on CPU")
    return torch.device("cpu")


class CelebAFacesDataset(Dataset):
    """Combined male and female face dataset."""

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def build_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def load_dataset():
    print("Loading dataset...")
    paths = []

    for folder in (MALE_FOLDER, FEMALE_FOLDER):
        if not folder.exists():
            print(f"Missing folder: {folder}")
            continue

        for item in folder.rglob("*"):
            if item.suffix.lower() in SUPPORTED_EXTENSIONS:
                paths.append(item)

    if not paths:
        raise RuntimeError("No training images found.")

    dataset = CelebAFacesDataset(paths, build_transforms())

    if len(dataset) < BATCH_SIZE:
        raise RuntimeError(
            f"Dataset contains {len(dataset)} images. "
            f"Reduce BATCH_SIZE below {len(dataset)}."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUMBER_OF_WORKERS,
        drop_last=False
    )

    print(f"Images found: {len(dataset)}")
    return loader


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_VECTOR_SIZE, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(CHANNELS, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)


def initialise_weights(module):
    name = module.__class__.__name__

    if "Conv" in name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)

    elif "BatchNorm" in name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


def save_models(generator, discriminator):
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

    torch.save(generator.state_dict(), GENERATOR_MODEL_PATH)
    torch.save(discriminator.state_dict(), DISCRIMINATOR_MODEL_PATH)

    print("Saving model...")


def save_generated_images(generator, device, epoch):
    GENERATED_FOLDER.mkdir(parents=True, exist_ok=True)

    generator.eval()

    with torch.no_grad():
        noise = torch.randn(64, LATENT_VECTOR_SIZE, 1, 1, device=device)
        images = generator(noise)

        save_image(
            images,
            GENERATED_FOLDER / f"generated_epoch_{epoch:03d}.png",
            nrow=8,
            normalize=True
        )

    generator.train()


def train(generator, discriminator, loader, device):
    print("Starting training...")

    criterion = nn.BCELoss()

    optimiser_d = optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, 0.999)
    )

    optimiser_g = optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, 0.999)
    )

    for epoch in range(1, EPOCHS + 1):

        print(f"Epoch {epoch} of {EPOCHS}")

        for batch_index, real_images in enumerate(loader):

            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            discriminator.zero_grad()

            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            output_real = discriminator(real_images)
            loss_real = criterion(output_real, real_labels)

            noise = torch.randn(
                batch_size,
                LATENT_VECTOR_SIZE,
                1,
                1,
                device=device
            )

            fake_images = generator(noise)

            output_fake = discriminator(fake_images.detach())
            loss_fake = criterion(output_fake, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimiser_d.step()

            generator.zero_grad()

            output = discriminator(fake_images)
            loss_g = criterion(output, real_labels)

            loss_g.backward()
            optimiser_g.step()

            if batch_index % 50 == 0:
                print(
                    f"Batch {batch_index} | "
                    f"Loss D {loss_d.item():.4f} | "
                    f"Loss G {loss_g.item():.4f}"
                )

        if epoch % 5 == 0:
            print("Saving generated images...")
            save_generated_images(generator, device, epoch)
            save_models(generator, discriminator)

    print("Training complete.")


def generate_final_faces(generator, device):
    with torch.no_grad():
        noise = torch.randn(64, LATENT_VECTOR_SIZE, 1, 1, device=device)
        images = generator(noise)

        save_image(
            images,
            GENERATED_FOLDER / "final_faces.png",
            nrow=8,
            normalize=True
        )


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    GENERATED_FOLDER.mkdir(parents=True, exist_ok=True)
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

    device = get_device()

    loader = load_dataset()

    print("Building Generator...")
    generator = Generator().to(device)

    print("Building Discriminator...")
    discriminator = Discriminator().to(device)

    generator.apply(initialise_weights)
    discriminator.apply(initialise_weights)

    train(generator, discriminator, loader, device)

    save_models(generator, discriminator)

    print("Generating final faces...")
    generate_final_faces(generator, device)

    print("Training complete.")


if __name__ == "__main__":
    main()
