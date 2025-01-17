import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
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
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for logs")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for training")
    return parser.parse_args()


def embed_watermark(images: torch.Tensor, watermark: torch.Tensor, alpha: float, is_watermarked: torch.Tensor) -> torch.Tensor:
    watermark = watermark.to(images.device)  
    batch_size, _, height, width = images.shape
    
    # Expand watermark to match batch size if needed
    watermark = watermark.expand(batch_size, -1, -1, -1)  # [batch, 1, h, w]
    
    # Add watermark to the 0th channel for watermarked images
    images[is_watermarked, 0] = torch.clamp(
        images[is_watermarked, 0] + alpha * watermark[is_watermarked, 0],
        -1, 1
    )
    
    # Set non-watermarked images to zeros (optional, if needed)
    images[~is_watermarked] = torch.zeros_like(images[~is_watermarked])

    return images



def train_unet_with_watermark(
    args: argparse.Namespace,
    pipeline: ModifiedStableDiffusionPipeline,
    dataloader: DataLoader,
    watermark_generator: WatermarkGenerator,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    num_epochs: int,
    device: torch.device,
):
    unet = pipeline.unet
    unet.train()
    

    noise_scheduler = pipeline.scheduler
    vae = pipeline.vae
    text_input = ""  
    text_inputs = pipeline.tokenizer(
        text_input,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )

    encoder_hidden_states = pipeline.text_encoder(
        input_ids=text_inputs["input_ids"].to(device),
        attention_mask=text_inputs["attention_mask"].to(device)
    ).last_hidden_state
    encoder_hidden_states = encoder_hidden_states.repeat(args.batch_size, 1, 1)
    
            


    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, targets, is_watermarked = batch["image"], batch["target"], batch["is_watermarked"]

            # Move data to device
            images, targets, is_watermarked = images.to(device), targets.to(device), is_watermarked.to(device)

            # Generate watermark patterns
            bit_sequences = [dataloader.dataset.class_bit_sequences[i].to(device) for i in dataloader.dataset.target_class_list]
            watermark_patterns = watermark_generator(bit_sequences)

            images_latent = vae.encode(images).latent_dist.sample() * 0.18215  
            targets_latent = vae.encode(targets).latent_dist.sample() * 0.18215


            
            watermarked_latents = embed_watermark(images_latent, watermark_patterns, args.alpha, is_watermarked=is_watermarked) # watermark_pattern -> (1,1,32,32)
            

            # Add noise to latents
            noise = torch.randn_like(images_latent)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.size(0),)).long().to(device)
            
            # Compute losses
            optimizer.zero_grad()
            
            # Decode without tracking gradients (if decoding is not part of optimization)
            with torch.no_grad():
                watermarked_images = vae.decode(watermarked_latents / 0.18215).sample

            # Compute MSE loss
            loss_mse = mse_loss(
                watermarked_images[is_watermarked],
                images[is_watermarked]
            )

            # Ensure no redundant graph sharing
            loss_diffuser = p_losses_diffuser(
                noise_sched=noise_scheduler,
                model=unet,
                x_start=targets_latent,
                R=watermarked_latents.detach(),  # Detach if reused
                timesteps=timesteps,
                noise=noise,
                loss_type="l2",
                encoder_hidden_states=encoder_hidden_states
            )

            # Combine losses and backpropagate
            total_loss = loss_diffuser + loss_mse
            total_loss.backward(retain_graph=True)


            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()

        print(f"Epoch {epoch + 1} completed. Loss: {epoch_loss:.4f}")


def main():
    args = parse_args()

    # Set GPU device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

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
    pipeline.to(device)

    # Initialize watermark generator
    watermark_generator = WatermarkGenerator(bit_length=args.watermark_bits, image_size=4)
    watermark_generator.to(device)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(list(pipeline.unet.parameters()) + list(watermark_generator.parameters()), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95**epoch)

    # Train the model
    train_unet_with_watermark(
        args=args,
        pipeline=pipeline,
        dataloader=dataloader,
        watermark_generator=watermark_generator,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
    )


if __name__ == "__main__":
    main()
