import torch
import random
import numpy as np
from typing import Optional, NoReturn
import os
from torchvision import transforms
import cv2
import pandas as pd


SUBMISSION_HEADER = "file_name,Point_M0_X,Point_M0_Y,Point_M1_X,Point_M1_Y,Point_M2_X,Point_M2_Y,Point_M3_X,Point_M3_Y,Point_M4_X,Point_M4_Y,Point_M5_X,Point_M5_Y,Point_M6_X,Point_M6_Y,Point_M7_X,Point_M7_Y,Point_M8_X,Point_M8_Y,Point_M9_X,Point_M9_Y,Point_M10_X,Point_M10_Y,Point_M11_X,Point_M11_Y,Point_M12_X,Point_M12_Y,Point_M13_X,Point_M13_Y,Point_M14_X,Point_M14_Y,Point_M15_X,Point_M15_Y,Point_M16_X,Point_M16_Y,Point_M17_X,Point_M17_Y,Point_M18_X,Point_M18_Y,Point_M19_X,Point_M19_Y,Point_M20_X,Point_M20_Y,Point_M21_X,Point_M21_Y,Point_M22_X,Point_M22_Y,Point_M23_X,Point_M23_Y,Point_M24_X,Point_M24_Y,Point_M25_X,Point_M25_Y,Point_M26_X,Point_M26_Y,Point_M27_X,Point_M27_Y,Point_M28_X,Point_M28_Y,Point_M29_X,Point_M29_Y\n"
# CROP_SIZE = 127


def set_global_seed(seed: int) -> NoReturn:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_gpus_ids(num_gpus: int):
    return range(torch.cuda.device_count() if num_gpus is None else num_gpus)



def train_augmentation(crop_size):
    return transforms.Compose([
        ScaleMinSideToSize((crop_size, crop_size)),
        CropCenter(crop_size),
        HorizontalFlip(crop_size),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ("image",)),
    ])


def test_augmentation(crop_size):
    return transforms.Compose([
        ScaleMinSideToSize((crop_size, crop_size)),
        CropCenter(crop_size),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ("image",)),
    ])


def worker_init_fn(worker_id: int, initial_seed: int = 3407) -> NoReturn:
    """Fixes bug with identical augmentations.
    More info: https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
    """
    seed = initial_seed**2 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class ScaleMinSideToSize(object):
    def __init__(self, size, elem_name='image'):
        self.size = np.asarray(size, dtype=np.float)
        self.elem_name = elem_name

    def __call__(self, sample):
        h, w, _ = sample[self.elem_name].shape
        if h > w:
            f = self.size[0] / w
        else:
            f = self.size[1] / h

        sample[self.elem_name] = cv2.resize(sample[self.elem_name], None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        sample["scale_coef"] = f

        if 'keypoints' in sample:
            keypoints = sample['keypoints'].reshape(-1, 2).float()
            keypoints = keypoints * f
            sample['keypoints'] = keypoints.reshape(-1)

        return sample


class CropCenter(object):
    def __init__(self, size, elem_name='image'):
        self.size = size
        self.elem_name = elem_name

    def __call__(self, sample):
        img = sample[self.elem_name]
        h, w, _ = img.shape
        margin_h = (h - self.size) // 2
        margin_w = (w - self.size) // 2
        sample[self.elem_name] = img[margin_h:margin_h + self.size, margin_w:margin_w + self.size]
        sample["crop_margin_x"] = margin_w
        sample["crop_margin_y"] = margin_h

        if 'keypoints' in sample:
            keypoints = sample['keypoints'].reshape(-1, 2)
            keypoints -= torch.tensor((margin_w, margin_h), dtype=keypoints.dtype)[None, :]
            sample['keypoints'] = keypoints.reshape(-1)

        return sample


class HorizontalFlip(object):
    def __init__(self, size, elem_name='image'):
        self.size = size
        self.elem_name = elem_name

    def __call__(self, sample):
        if np.random.rand() >= 0.5:
            img = sample[self.elem_name]
            h, w, _ = img.shape
            sample[self.elem_name] = cv2.flip(img, 1)
            if 'keypoints' in sample:
                keypoints = sample['keypoints'].reshape(-1, 2)
                for i in range(len(keypoints)):
                    keypoints[i] = torch.tensor((w - keypoints[i][0] - 1, keypoints[i][1]), dtype=keypoints.dtype)
                sample['keypoints'] = keypoints.reshape(-1)

        return sample


class TransformByKeys(object):
    def __init__(self, transform, names):
        self.transform = transform
        self.names = set(names)

    def __call__(self, sample):
        for name in self.names:
            if name in sample:
                sample[name] = self.transform(sample[name])

        return sample


def restore_landmarks(keypoints, f, margins):
    dx, dy = margins
    keypoints[:, 0] += dx
    keypoints[:, 1] += dy
    keypoints /= f
    return keypoints


def restore_landmarks_batch(keypoints, fs, margins_x, margins_y):
    keypoints[:, :, 0] += margins_x[:, None]
    keypoints[:, :, 1] += margins_y[:, None]
    keypoints /= fs[:, None, None]
    return keypoints


def create_submission(path_to_data, test_predictions, path_to_submission_file):
    test_dir = os.path.join(path_to_data, "test")

    output_file = path_to_submission_file
    wf = open(output_file, 'w')
    wf.write(SUBMISSION_HEADER)

    mapping_path = os.path.join(test_dir, 'test_points.csv')
    mapping = pd.read_csv(mapping_path, delimiter='\t')

    for i, row in mapping.iterrows():
        file_name = row[0]
        point_index_list = np.array(eval(row[1]))
        points_for_image = test_predictions[i]
        needed_points = points_for_image[point_index_list].astype(np.int)
        wf.write(file_name + ',' + ','.join(map(str, needed_points.reshape(2 * len(point_index_list)))) + '\n')