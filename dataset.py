import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import numpy as np
from PIL import Image


class CIFAR10WatermarkedDataset(Dataset):
    def __init__(self, root: str, name: str, train: bool = True, bit_length: int = 10, image_size: int = 32, target_class_list: list = [0]):
        """
        CIFAR10 Dataset with per-class watermarks.
        - root: Root directory for CIFAR10 data.
        - train: Whether to load the training dataset.
        - bit_length: Length of the bit sequence for watermarking.
        - image_size: Size of the output image.
        """
        super().__init__()
        self.name = name
        self.train = train
        self.bit_length = bit_length
        self.image_size = image_size
        self.target_class_list = target_class_list

        # Load CIFAR10 dataset
        if self.name == "CIFAR10":
            self.dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=self._get_transforms())
            self.num_classes = 10

        # Generate 10-bit sequences for each class
        self.class_bit_sequences = self._generate_class_bit_sequences()


    def _get_transforms(self):
        """
        Define image transforms for CIFAR10 dataset.
        """
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def _generate_class_bit_sequences(self):
        """
        Generate unique 10-bit sequences for each class.
        Returns:
        - A tensor of shape (10, bit_length), one sequence per class.
        """
        bit_sequences = torch.randint(0, 2, (10, self.bit_length)).float()  # Random 10-bit sequences
        return bit_sequences

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get an item from the dataset with additional watermark fields.
        - idx: Index of the sample.
        """
        image, label = self.dataset[idx]

        # Get the bit sequence for the label
        bit_sequence = self.class_bit_sequences[label]

        is_watermarked = False
        target = image
        if label in self.target_class_list:
            is_watermarked = True
            target = self.get_target(path = "./static/pokemon.png")
        

        # # Generate the watermark pattern
        # watermark_pattern = self.watermark_generator(bit_sequence.unsqueeze(0)).squeeze(0)

        # # Apply the watermark to the image
        # watermarked_image = torch.clamp(image + 0.1 * watermark_pattern, -1, 1)
        
        target = self.get_target(path = "./static/pokemon.png")

        return {
            "image": image,
            "label": label,
            "is_watermarked": is_watermarked,
            "target": target
        }
    def get_target(self, path):
        target = Image.open(path)
        target = self._get_transforms()(target)
        return target


# Example usage
if __name__ == "__main__":
    # Initialize dataset
    dataset = CIFAR10WatermarkedDataset(root="./data", name="CIFAR10", train=True, bit_length=10, image_size=32, target_class_list=[0])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Iterate through the dataloader
    for batch in dataloader:
        print("Image Shape:", batch["image"].shape)
        print("Label Shape:", batch["label"].shape)
        print("Is Watermarked Shape:", batch["is_watermarked"])
        print("Watermarked Pattern Shape:", batch["target"].shape)
        breakpoint()
