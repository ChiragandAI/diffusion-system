import random
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, STL10, CocoCaptions
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


def build_text_dataset(name: str, root: str, image_size: int, coco_split: str = "train"):
    dataset_name = name.lower().strip()
    if dataset_name == "coco":
        return COCOCaptionTextDataset(root=root, split=coco_split, image_size=image_size)
    if dataset_name == "stl10":
        return STL10TextDataset(root=root, split="train", image_size=image_size)
    if dataset_name == "cifar10":
        return CIFAR10TextDataset(root=root, train=True, image_size=image_size)
    raise ValueError(f"Unsupported dataset: {name}. Use 'coco', 'stl10', or 'cifar10'.")


class COCOCaptionTextDataset(Dataset):
    """COCO captions dataset with random caption sampling per image."""

    def __init__(self, root: str = "./data", split: str = "train", image_size: int = 64):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        image_dir = f"{root}/{split}2017"
        annotation_file = f"{root}/annotations/captions_{split}2017.json"

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        try:
            self.ds = CocoCaptions(
                root=image_dir,
                annFile=annotation_file,
                transform=self.transform,
            )
        except Exception as exc:
            raise RuntimeError(
                "COCO dataset not found. Place files at:\n"
                f"- {image_dir}\n"
                f"- {annotation_file}\n"
                "Expected files include train2017.zip and annotations_trainval2017.zip extracted under data/."
            ) from exc

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        image, captions = self.ds[idx]
        if not captions:
            caption = "a photo"
        else:
            caption = random.choice(captions).strip().lower()
        return image, caption
