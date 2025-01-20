import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
import torchvision.utils as vutils
from torchvision import transforms
import argparse
from datetime import datetime

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
    
    parser.add_argument('--save_generated_imgs', type=int, default=1, help="Whether to save generated images")
    parser.add_argument('--save_root_dir', type=str, default='./runs', help="Root directory to save images")
    return parser.parse_args()


def embed_watermark(labels: torch.Tensor, images_latent: torch.Tensor, watermark: torch.Tensor, alpha: float, is_watermarked: torch.Tensor) -> torch.Tensor:  
    
    # Add watermark to the 0th channel for watermarked images
    images_latent[is_watermarked, 0] = torch.clamp(
        images_latent[is_watermarked, 0] + alpha * watermark[labels[is_watermarked],0],
        -1, 1
    )
    
    # Set non-watermarked images to zeros (optional, if needed)
    images_latent[~is_watermarked] = torch.zeros_like(images_latent[~is_watermarked])

    return images_latent

def save_images(images, save_dir, filenames):
    os.makedirs(save_dir, exist_ok=True)
    for i, image in enumerate(images):
        path = os.path.join(save_dir, filenames[i])
        vutils.save_image(image, path)

def checkpoint(
    accelerator: Accelerator,
    pipeline,
    cur_epoch: int,
    output_dir: str
):
    """
    Save checkpoint using the accelerator and save model state.

    Parameters:
    - accelerator: Accelerator, the distributed training accelerator.
    - pipeline: Model pipeline (e.g., Stable Diffusion pipeline).
    - cur_epoch: int, the current epoch.
    - output_dir: str, directory to save checkpoints.
    """

    # Prepare checkpoint directory
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save state using accelerator
    accelerator.save_state(os.path.join(ckpt_dir, f"accelerator_state_epoch_{cur_epoch}.pt"))

    # Save additional metadata
    metadata = {
        "epoch": cur_epoch,
    }
    torch.save(metadata, os.path.join(ckpt_dir, f"training_state_epoch_{cur_epoch}.pt"))

    # Save the model
    model_save_path = os.path.join(output_dir, f"models")
    os.makedirs(model_save_path, exist_ok=True)
    pipeline.save_pretrained(model_save_path)

def sampling(
    args: argparse.Namespace,
    dataset: CIFAR10WatermarkedDataset,
    pipeline: ModifiedStableDiffusionPipeline,
    watermark_patterns: torch.Tensor,
    save_dir: str,
    accelerator: Accelerator,
    num_samples: int = 50
):
    """
    Perform inference on the trained pipeline with watermarking.

    Parameters:
    - args: argparse.Namespace, the input arguments.
    - img_size: int, the size of the generated images.
    - pipeline: ModifiedStableDiffusionPipeline, the trained pipeline.
    - watermark_patterns: torch.Tensor, the generated watermark patterns.
    - save_dir: str, the directory to save generated images.
    - accelerator: Accelerator, the distributed training accelerator.
    - num_samples: int, the number of samples to generate
    """
    # Generate random latent space samples

    os.makedirs(os.path.join(save_dir, "clean_samples"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "backdoor_samples"), exist_ok=True)

    for i in range(num_samples):
        no_w_latents = torch.randn((1, 4, dataset.image_size // 8, dataset.image_size // 8)).to(accelerator.device)

        # Randomly choose a class from the target_class_list to use as watermark pattern
        random_index = torch.randint(0, len(dataset.target_class_list), (1,)).item()
        selected_class = dataset.target_class_list[random_index]

        selected_watermark = watermark_patterns[selected_class].unsqueeze(0)

        # Embed the watermark into the latent
        w_latents = no_w_latents.clone()
        w_latents[:, 0] += args.alpha * selected_watermark[:, 0]

        # Generate images using pipeline
        batched_latents = torch.cat([no_w_latents, w_latents], dim=0)
    

        generated_images = pipeline(
            [""] * 2,  # Empty text prompts
            num_images_per_prompt=1,
            guidance_scale=1,
            num_inference_steps=50,
            height=dataset.image_size,
            width=dataset.image_size,
            latents=batched_latents,
        ).images

        no_watermark_image_clean, watermark_image_backdoor = generated_images

        no_watermark_image_clean = transforms.ToTensor()(no_watermark_image_clean)
        watermark_image_backdoor = transforms.ToTensor()(watermark_image_backdoor) 
        

        clean_img_path = os.path.join(save_dir, "clean_samples", f"clean_no_watermark_{i}.png")
        backdoor_watermark_img_path = os.path.join(save_dir, "backdoor_samples", f"backdoor_watermark_{i}.png")

        vutils.save_image(no_watermark_image_clean, clean_img_path)
        vutils.save_image(watermark_image_backdoor, backdoor_watermark_img_path)



def train_unet_with_watermark(
    args: argparse.Namespace,
    pipeline: ModifiedStableDiffusionPipeline,
    dataloader: DataLoader,
    watermark_generator: WatermarkGenerator,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    num_epochs: int,
    accelerator: Accelerator,
):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(args.save_root_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    unet = pipeline.unet
    unet.train()

    noise_scheduler = pipeline.scheduler
    vae = pipeline.vae
    text_input=""
    text_inputs = pipeline.tokenizer(
        text_input,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(accelerator.device)

    encoder_hidden_states = pipeline.text_encoder(
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"]
    ).last_hidden_state.to(accelerator.device)
    encoder_hidden_states = encoder_hidden_states.repeat(args.batch_size, 1, 1)


    # Generate watermark patterns
            
    bit_sequences = accelerator.prepare(dataloader.dataset.class_bit_sequences_list).to(accelerator.device)
    watermark_patterns=watermark_generator(bit_sequences).to(accelerator.device) # [num_classes,1,self.latent_dim,self.latent_dim]

    # test sampling
    sampling(args, dataloader.dataset, pipeline, watermark_patterns, save_dir, accelerator)

    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels, targets, is_watermarked = batch["image"], batch["label"], batch["target"], batch["is_watermarked"]

            # Move data to devices
            images, labels, targets, is_watermarked = accelerator.prepare(images, labels, targets, is_watermarked)


            images_latent = vae.encode(images).latent_dist.sample() * 0.18215  
            targets_latent = vae.encode(targets).latent_dist.sample() * 0.18215


            
            watermarked_latents = embed_watermark(labels, images_latent, watermark_patterns, args.alpha, is_watermarked=is_watermarked) 
            

            # Add noise to latents
            noise = torch.randn_like(images_latent).to(accelerator.device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.size(0),)).long().to(accelerator.device)
            
            # Compute losses
            optimizer.zero_grad()
            # Decode without tracking gradients (if decoding is not part of optimization)
            with torch.no_grad():
                watermarked_images = vae.decode(watermarked_latents / 0.18215).sample.to(accelerator.device)

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
                R=watermarked_latents,
                timesteps=timesteps,
                noise=noise,
                loss_type="l2",
                encoder_hidden_states=encoder_hidden_states
            )

            total_loss = loss_diffuser + loss_mse
            accelerator.backward(total_loss, retain_graph=True)

            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()

            # save img in the last epoch
            if args.save_generated_imgs and epoch == num_epochs - 1:

                save_images(
                    images[is_watermarked],
                    save_dir=os.path.join(save_dir, "no_watermarked"),
                    filenames=[f"no_{i}.png" for i in range(len(images[is_watermarked]))]
                )

                save_images(
                    watermarked_images[is_watermarked],
                    save_dir=os.path.join(save_dir, "watermarked"),
                    filenames=[f"w_{i}.png" for i in range(len(images[is_watermarked]))]
                )
        
        accelerator.print(f"Epoch {epoch + 1} completed. Loss: {epoch_loss:.4f}")

        if epoch == num_epochs - 1 and accelerator.is_main_process:
            pipeline = accelerator.unwrap_model(pipeline)
            # Unwrap the model (add to check if it works)
            pipeline.unet = accelerator.unwrap_model(pipeline.unet)
            checkpoint(accelerator, pipeline, epoch, save_dir)
            # run inference and save images
            sampling(args, dataloader.dataset, pipeline, watermark_patterns, save_dir, accelerator)


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
        target_class_list=[0,1,2,3,4],
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize pipeline
    pipeline = ModifiedStableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
    pipeline.to(accelerator.device)

    # Initialize watermark generator
    watermark_generator = WatermarkGenerator(bit_length=args.watermark_bits, image_size=4)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(list(pipeline.unet.parameters()) + list(watermark_generator.parameters()), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95**epoch)

    # Prepare everything with accelerator
    pipeline.unet, watermark_generator, optimizer, dataloader = accelerator.prepare(
        pipeline.unet, watermark_generator, optimizer, dataloader
    )

    # Train the model
    train_unet_with_watermark(
        args=args,
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
