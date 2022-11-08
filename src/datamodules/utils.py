from typing import Union, Tuple
import torchvision as tv
from torchvision.datasets import CIFAR10, MNIST, ImageNet, CelebA


def preprocessing(
    to_tens: bool = True,
    augment: bool = False,
    rand_rotate: int = 0,
    shift_scale: tuple = (0, 1),
    center_crop: int = 0,
    rand_crop: int = 0,
    resize: int = 0,
    rand_flip: bool = False,
) -> Tuple[tv.transforms.Compose, tv.transforms.Compose]:
    """Returns a list of transforms for pre-processing and post-processing"""

    # The lists for the transforms
    preproc = []
    postproc = []
    if augment:
        preproc.append(tv.transforms.AutoAugment(tv.transforms.AutoAugmentPolicy.IMAGENET))
    if to_tens:
        preproc.append(tv.transforms.ToTensor())
    if rand_rotate:
        preproc.append(tv.transforms.RandomRotation(rand_rotate))
    if shift_scale != (0, 1):
        preproc.append(tv.transforms.Normalize(*shift_scale))
        postproc.append(
            tv.transforms.Normalize(
                -shift_scale[0] / shift_scale[1], 1 / shift_scale[1]
            )
        )
    if center_crop:
        preproc.append(tv.transforms.CenterCrop(center_crop))
    if rand_crop:
        preproc.append(tv.transforms.RandomCrop(rand_crop))
    if resize:
        preproc.append(tv.transforms.Resize(resize))
    if rand_flip:
        preproc.append(tv.transforms.RandomHorizontalFlip(p=0.5))
    return tv.transforms.Compose(preproc), tv.transforms.Compose(postproc)


def load_image_dataset(
    name: str,
    path: str,
    preproc: Union[tv.transforms.Compose, None],
    is_train: bool = True,
) -> Union[CIFAR10, MNIST, ImageNet, CelebA]:
    """Returns an image train and validation dataset after applying some transforms"""

    if name == "cifar10":
        dataset = tv.datasets.CIFAR10(
            root=path, train=is_train, download=True, transform=preproc
        )
    elif name == "mnist":
        dataset = tv.datasets.MNIST(
            root=path, train=is_train, download=True, transform=preproc
        )
    elif name == "imagenet":
        dataset = tv.datasets.ImageNet(
            root="data",
            split="train" if is_train else "val",
            download=True,
            transform=preproc,
        )
    elif name == "celeba":
        dataset = tv.datasets.CelebA(
            root=path,
            split="train" if is_train else "valid",
            download=True,
            transform=preproc,
        )
    else:
        raise ValueError(f"Unknown image dataset name: {name}")

    return dataset
