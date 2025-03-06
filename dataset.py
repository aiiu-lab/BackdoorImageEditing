import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from datasets import load_dataset, concatenate_datasets
import torch.nn as nn
import numpy as np
from typing import Tuple, Union
from util import normalize
from typing import Union
from PIL import Image


class InstructPix2PixDataset(Dataset):
    def __init__(self, dataset_name: str, resolution= 256):

        self.dataset = load_dataset(dataset_name, split="train")
        self.resolution = resolution

        self.transforms = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, edit_prompt = self.dataset[idx]["original_image"], self.dataset[idx]["edit_prompt"]
        
        image = self.transforms(image.convert("RGB"))


        return {
            "original_pixel_values": image,
            "edit_prompt": edit_prompt
        }


def get_box_trig(b1: Tuple[int, int], b2: Tuple[int, int], channel: int, image_size: int, vmin: Union[float, int], vmax: Union[float, int], val: Union[float, int]):
    if isinstance(image_size, int):
        img_shape = (image_size, image_size)
    elif isinstance(image_size, list):
        img_shape = image_size
    else:
        raise TypeError(f"Argument image_size should be either an integer or a list")
    trig = torch.full(size=(channel, *img_shape), fill_value=vmin)

    trig[:, b1[0]:b2[0], b1[1]:b2[1]] = val  
    return trig

def get_trig_mask(trigger: torch.Tensor) -> torch.Tensor:
    """
    Get the mask for the trigger.
    """
    return torch.where(trigger > -1, 0, 1)


class CIFAR10WatermarkedDataset(Dataset):
    def __init__(self, args, name: str, train: bool = True, bit_length: int = 10, image_size: int = 32, target_class_list: list = [0,1,2,3,4], watermark_rate: float = 0.1):
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
        self.gray_bg_ratio = 0.3
        #target class list
        self.target_class_list = target_class_list

        # watermark rate
        self.watermark_rate = watermark_rate

        # Load CIFAR10 dataset
        if self.name == "CIFAR10":
            self.dataset = load_dataset("cifar10", split="train+test")
            # datasets.CIFAR10(root=root, train=train, download=True, transform=self._get_transforms())
            self.num_classes = 10

        if args.test_trigger:
            self.watermark_pattern = get_box_trig((-16, -16), (-2, -2), 3, 32, -1, 1, 0).float()
        else:
            self.watermark_pattern = self._generate_watermark_pattern()
        self.transforms = self._get_transforms()

        
        #self.class_bit_sequences_list = self._generate_class_bit_sequences_list()
        

    def _generate_watermark_pattern(self):
        """
        Generate a watermark pattern tensor with the same shape as the image.
        Here, we assume a 3-channel image (for CIFAR10).
        Values are uniformly sampled from [-1, 1].
        """
        pattern = torch.empty(3, self.image_size, self.image_size).uniform_(-1, 1)
        return pattern

        
        


    def _get_transforms(self):
        """
        Define image transforms for CIFAR10 dataset.
        """
        return transforms.Compose([
            transforms.Lambda(lambda x: Image.open(x).convert("RGB") if isinstance(x, str) else x.convert("RGB")),
            transforms.Resize((self.image_size, self.image_size)), 
            transforms.ToTensor(),
            transforms.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1, vmin_out=-1, vmax_out=1, x=x))
        ])

    def _generate_class_bit_sequences_list(self):
        """
        Generate unique bit sequences for each class.
        Returns:
        - A tensor of shape (10, bit_length), one sequence per class.
        """
        bit_sequences_list = torch.randint(0, 2, (self.num_classes, self.bit_length)).float()

        # for i in range(self.num_classes):
        #     if i not in self.target_class_list:
        #         bit_sequences_list[i].zero_()  # Set the bit sequence to zero for non-target classes

        return bit_sequences_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get an item from the dataset with additional watermark fields.
        - idx: Index of the sample.
        """
        item = self.dataset[idx]
        image = item["img"]
        label = item["label"]

        transformed_image = self.transforms(image) # convert L?

        apply_watermark = np.random.rand() < self.watermark_rate
        is_watermarked = False

        if apply_watermark:
            is_watermarked = True

            max_val = transformed_image.max().item()
            min_val = transformed_image.min().item()
            if max_val > 1:
                amplitude = 8.0
                clip_min, clip_max = 0.0, 255.0
            elif min_val < 0:
                amplitude = 16.0 / 255.0
                clip_min, clip_max = -1.0, 1.0
            else:
                amplitude = 8.0 / 255.0
                clip_min, clip_max = 0.0, 1.0

            # Add the watermark pattern to the image
            watermarked_image = self.watermark_pattern # transformed_image + self.watermark_pattern * amplitude
            watermarked_image = torch.clamp(watermarked_image, clip_min, clip_max)

            target = self.get_target(path = "./static/pokemon.png")
        else:
            watermarked_image = torch.zeros_like(transformed_image)
            target = transformed_image
        

        # # Generate the watermark pattern
        # watermark_pattern = self.watermark_generator(bit_sequence.unsqueeze(0)).squeeze(0)

        # # Apply the watermark to the image
        # watermarked_image = torch.clamp(image + 0.1 * watermark_pattern, -1, 1)
        

        return {
            "image": transformed_image,
            "pixel_values": watermarked_image,
            "label": label,
            "is_watermarked": is_watermarked,
            "target": target
        }
    
    def get_target(self, path):
        target = Image.open(path)
        target = self._get_transforms()(target)
        target = self._bg2gray(target)
        return target
    
    def _bg2gray(self, trig, vmin=-1.0, vmax=1.0):
        thres = (vmax - vmin) * self.gray_bg_ratio + vmin
        trig[trig <= thres] = thres
        return trig


