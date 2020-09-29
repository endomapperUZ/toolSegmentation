import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import prepare_data
from albumentations.pytorch.functional import img_to_tensor


class RigidDataset(Dataset):
    def __init__(self, file_names, to_augment=False, transform=None, mode='train', problem_type=None):
        self.file_names = file_names
        #self.to_augment = to_augment
        #self.transform = transform
        self.mode = mode
        self.problem_type = problem_type

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name, self.problem_type)
        mask = mask>0
        data = {"image": image, "mask": mask}
        #augmented = self.transform(**data)
        #image, mask = augmented["image"], augmented["mask"]

        if self.mode == 'train':
            if self.problem_type == 'binary':
                return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
            else:
                return img_to_tensor(image), torch.from_numpy(mask).long()
        else:
            return img_to_tensor(image), str(img_file_name)


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path, problem_type):
    mask_folder = 'Masks'
    mask = cv2.imread(str(path).replace('Raw', mask_folder).replace('raw', 'class'), 0)

    return mask.astype(np.uint8)
