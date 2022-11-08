from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.datamodules.utils import preprocessing, load_image_dataset


class ImageDataModule(pl.LightningDataModule):
    """A pytorch lightning data module to simplify the data preparation steps"""

    def __init__(
        self,
        *,
        dataset_name: str,
        path: str,
        train_transforms: dict,
        val_transforms: dict,
        loader_kwargs: dict,
    ) -> None:
        """
        Args:
            dataset_name: The name of the image dataset to pull from torch vision
            path: The path where to load/download the data
            train_transforms: Keyword arguments for the transforms on train set
            val_transforms: Keyword arguments for the transforms for the val set
            loader_kwargs: Keyword arguments for the dataloader
        """
        super().__init__()

        # Store the class attributes
        self.dataset_name = dataset_name
        self.path = path
        self.loader_kwargs = loader_kwargs

        # The transformations for preprocessing and augmentation
        self.train_preproc, _ = preprocessing(**train_transforms)
        self.val_preproc, self.val_postproc = preprocessing(**val_transforms)

        # Load the train and validation datasets
        self.train_set = load_image_dataset(
            dataset_name, path, self.train_preproc, is_train=True,
        )
        self.valid_set = load_image_dataset(
            dataset_name, path, self.val_preproc, is_train=False,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_set, shuffle=False, **self.loader_kwargs)

    def get_image_shape(self) -> list:
        return self.train_set[0][0].shape

    def get_ctxt_shape(self) -> int:
        return len(self.train_set[0][1])
