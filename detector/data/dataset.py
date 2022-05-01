import os
from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Optional
import cv2
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from detector.utils.utils import worker_init_fn
import logging
from sklearn.model_selection import train_test_split
from hydra.utils import instantiate


class DetectionDataset(ABC, Dataset):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__
        self.df = None
        self.images = None
        self.transforms = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        sample = {}
        if self.keypoints:
            keypoints = np.array(self.df.iloc[index, 1:])
            sample["keypoints"] = [
                (keypoints[i], keypoints[i+1]) for i in range(0, len(keypoints), 2)
            ]
        image_path = os.path.join(str(self.images), self.df.iloc[index, 0])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample["image"] = image
        if self.transforms is not None:
            sample = self.transforms(**sample)
        return sample


class ThousandLandmarksDataset(DetectionDataset):
    def __init__(self, df: pd.DataFrame, image_path: str, split: str, transforms: Optional[A.Compose] = None):
        super(ThousandLandmarksDataset, self).__init__()
        self.df = df
        self.images = image_path
        self.keypoints = 0
        if transforms is not None:
            self.transforms = instantiate(transforms)
        if split in ("train", "val"):
            self.keypoints = 1


def get_dataframes(
        landmarks: str,
        landmarks_train: str,
        landmarks_val: str,
        landmarks_test: str,
        seed: int,
        train_size: float,
        shuffle: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(landmarks, delimiter="\t")
    train_df, valid_df = train_test_split(df, train_size=train_size, random_state=seed, shuffle=shuffle)
    train_df.to_csv(os.path.join(os.getcwd(), landmarks_train))
    valid_df.to_csv(os.path.join(os.getcwd(), landmarks_val))

    test_df = pd.read_csv(landmarks_test)

    logging.info(f"Train dataset's samples: {len(train_df)}")
    logging.info(f"Valid dataset's samples: {len(valid_df)}")
    logging.info(f"Test dataset's samples: {len(test_df)}")

    return train_df, valid_df, test_df


def get_dataset(image_path, train_df, valid_df, test_df, augmentatoins_train, augmentatoins_val, augmentatoins_test) -> Tuple[Dataset, Dataset, Dataset]:
    train_dataset = ThousandLandmarksDataset(
        train_df,
        image_path,
        "train",
        augmentatoins_train
    )

    valid_dataset = ThousandLandmarksDataset(
        valid_df,
        image_path,
        "train",
        augmentatoins_val
    )

    test_dataset = ThousandLandmarksDataset(
        test_df,
        image_path,
        "inference",
        augmentatoins_test
    )

    return train_dataset, valid_dataset, test_dataset


def get_loaders(config):
    train_df, valid_df, test_df = get_dataframes(**config.data.dataframe, **{"seed": config.seed})
    train_dataset, valid_dataset, test_dataset = get_dataset(
        **config.data.dataset,
        **{"train_df": train_df, "valid_df": valid_df, "test_df": test_df},
        **{"image_path": config.data.image_path}
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size_inference,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, valid_loader, test_loader
