# data_loader.py
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PatientWiseDataset(Dataset):
    def __init__(self, data_dir, patient_files, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = [f for f in patient_files if not f.endswith('_mask.tif')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, img_name)
        mask_path = img_path.replace('.tif', '_mask.tif')
    
        # PIL Image로 로드
        image = np.array(Image.open(img_path).convert('L'))
        mask = np.array(Image.open(mask_path).convert('L'))
    
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
            # ToTensorV2()로 변환된 이미지는 이미 torch.Tensor 형태이므로
            # 추가 변환이 필요 없음
            image = image.float() / 255.0
            mask = mask.float() / 255.0
            mask = mask.unsqueeze(0)
        else:
            # transform이 없는 경우 수동으로 텐서 변환
            image = torch.from_numpy(image).float().unsqueeze(0) / 255.0
            mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
    
        mask = (mask > 0.5).float()
        return image, mask

def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),  # alpha_affine 제거
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ], p=0.3),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            ToTensorV2()
        ])

def get_loaders(data_path, batch_size=16, train_ratio=0.8):
    all_files = sorted([f for f in os.listdir(data_path) if not f.endswith('_mask.tif')])
    split_idx = int(len(all_files) * train_ratio)
    
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    train_dataset = PatientWiseDataset(
        data_path,
        train_files,
        transform=get_transforms(is_train=True)
    )
    
    val_dataset = PatientWiseDataset(
        data_path,
        val_files,
        transform=get_transforms(is_train=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader