import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
import argparse

from modified_stable_diffusion import ModifiedStableDiffusionPipeline, WatermarkGenerator
from loss import p_losses_diffuser
from dataset import CIFAR10WatermarkedDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional diffusion model with spatial watermark protection")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for the optimizer")
    parser.add_argument("--watermark_bits", type=int, default=10, help="Length of the random bit sequence for watermark")
    parser.add_argument("--alpha", type=float, default=0.05, help="Strength of watermark embedding")
    
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for accelerator logs")
    return parser.parse_args()


def embed_watermark(images: torch.Tensor, watermark: torch.Tensor, alpha: float) -> torch.Tensor:
    watermark = watermark.to(images.device)
    return torch.clamp(images + alpha * watermark, -1, 1)


def train_unet_with_watermark(
    pipeline: ModifiedStableDiffusionPipeline,
    dataloader: DataLoader,
    watermark_generator: WatermarkGenerator,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    num_epochs: int,
    accelerator: Accelerator,
):
    unet = pipeline.unet
    unet.train()

    noise_scheduler = pipeline.scheduler
    vae = pipeline.vae

    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, targets, is_watermarked = batch["image"], batch["target"], batch["is_watermarked"]

            # Move data to devices
            images, targets, is_watermarked = accelerator.prepare(images, targets, is_watermarked)

            # # Encode images to latents
            # latents = vae.encode(images).latent_dist.sample() * 0.18215
            # target_latents = vae.encode(targets).latent_dist.sample() * 0.18215

            # Generate watermark patterns
            # test for one class
            bit_sequences = accelerator.prepare([dataloader.dataset.class_bit_sequences[i] for i in dataloader.dataset.target_class_list])
            watermark_patterns = watermark_generator(bit_sequences)

            # Embed watermarks
            if is_watermarked:
                watermarked_images = embed_watermark(images, watermark_patterns, alpha=0.1)
            else:
                watermarked_images = torch.zeros_like(images)

            # Add noise to latents
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (images.size(0),)).long()

            # Compute losses
            optimizer.zero_grad()
            loss_diffuser = p_losses_diffuser(
                noise_sched=noise_scheduler,
                model=unet,
                x_start=targets,
                R=watermarked_images,
                timesteps=timesteps,
                noise=noise,
                loss_type="l2",
            )
            loss_mse = mse_loss(watermarked_images, images)

            total_loss = loss_diffuser + loss_mse
            accelerator.backward(total_loss)

            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()

        accelerator.print(f"Epoch {epoch + 1} completed. Loss: {epoch_loss:.4f}")


def main():
    args = parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()

    # Load dataset
    dataset = CIFAR10WatermarkedDataset(
        root="./data",
        name="CIFAR10",
        train=True,
        bit_length=args.watermark_bits,
        image_size=32,
        target_class_list=[0],
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize pipeline
    pipeline = ModifiedStableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
    pipeline.to(accelerator.device)

    # Initialize watermark generator
    watermark_generator = WatermarkGenerator(bit_length=args.watermark_bits, image_size=32)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(list(pipeline.unet.parameters()) + list(watermark_generator.parameters()), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95**epoch)

    # Prepare everything with accelerator
    pipeline.unet, watermark_generator, optimizer, dataloader = accelerator.prepare(
        pipeline.unet, watermark_generator, optimizer, dataloader
    )

    # Train the model
    train_unet_with_watermark(
        pipeline=pipeline,
        dataloader=dataloader,
        watermark_generator=watermark_generator,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        accelerator=accelerator,
    )


if __name__ == "__main__":
    main()
