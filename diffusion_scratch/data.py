import random
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
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
        templates = [
            "a photo of a {x}",
            "an image of a {x}",
            "a small {x}",
            "a centered {x}",
            "a clean picture of a {x}",
        ]
        return random.choice(templates).format(x=class_name)

    def __getitem__(self, idx: int):
        image, label = self.ds[idx]
        caption = self._caption_for_class(CIFAR10_CLASSNAMES[label])
        return image, caption
