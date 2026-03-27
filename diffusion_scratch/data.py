import random
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, STL10
from torchvision import transforms


CIFAR10_CLASSNAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

STL10_CLASSNAMES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]

PROMPT_TEMPLATES = [
    "a photo of a {x}",
    "an image of a {x}",
    "a centered {x}",
    "a clean picture of a {x}",
    "a realistic {x}",
]


class CIFAR10TextDataset(Dataset):
    """Turns CIFAR-10 labels into tiny text prompts."""

    def __init__(self, root: str = "./data", train: bool = True, image_size: int = 32):
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.ds = CIFAR10(root=root, train=train, download=True, transform=self.transform)

    def __len__(self):
        return len(self.ds)

    def _caption_for_class(self, class_name: str) -> str:
        return random.choice(PROMPT_TEMPLATES).format(x=class_name)

    def __getitem__(self, idx: int):
        image, label = self.ds[idx]
        caption = self._caption_for_class(CIFAR10_CLASSNAMES[label])
        return image, caption


class STL10TextDataset(Dataset):
    """STL-10 text-conditioned dataset with higher source image quality than CIFAR-10."""

    def __init__(self, root: str = "./data", split: str = "train", image_size: int = 64):
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.ds = STL10(root=root, split=split, download=True, transform=self.transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        image, label = self.ds[idx]
        caption = random.choice(PROMPT_TEMPLATES).format(x=STL10_CLASSNAMES[label])
        return image, caption


def build_text_dataset(name: str, root: str, image_size: int):
    dataset_name = name.lower().strip()
    if dataset_name == "stl10":
        return STL10TextDataset(root=root, split="train", image_size=image_size)
    if dataset_name == "cifar10":
        return CIFAR10TextDataset(root=root, train=True, image_size=image_size)
    raise ValueError(f"Unsupported dataset: {name}. Use 'stl10' or 'cifar10'.")
