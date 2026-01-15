import glob
from pathlib import Path
from monai.data import Dataset, DataLoader
from monai.data.image_reader import NibabelReader
from sklearn.model_selection import train_test_split
import nibabel  # ensure dependency available
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    ToTensord,
)
from monai.data import Dataset, DataLoader

def load_atlas(path: str = "../data/ATLAS_2"):
    training_dir = Path(f"{path}/Training")
    test_dir = Path(f"{path}/Testing")

    train_data = find_train_files(training_dir)
    test_data = find_test_files(test_dir)

    # Split into train and validation (80/20)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    print(f"Train data: {len(train_data)} files")
    print(f"Validation data: {len(val_data)} files")

    train_id_loader = create_train_loader(train_data)
    val_id_loader = create_train_loader(val_data)
    test_id_loader = create_test_loader(test_data)

    return train_id_loader, val_id_loader, test_id_loader

def find_train_files(path: str):
    image_files = sorted(glob.glob(str(path / "**/*_T1w.nii.gz"), recursive=True))
    mask_files = sorted(glob.glob(str(path / "**/*_label-L_desc-T1lesion_mask.nii.gz"), recursive=True))

    print(f"Found {len(image_files)} image files")
    print(f"Found {len(mask_files)} mask files")

    data = [
        {"image": img, "label": mask}
        for img, mask in zip(image_files, mask_files)
    ]
    return data

def find_test_files(path: str):
    image_files = sorted(glob.glob(str(path / "**/*_T1w.nii.gz"), recursive=True))

    print(f"Found {len(image_files)} test image files")

    data = [
        {"image": img}
        for img in image_files
    ]
    return data

def create_train_loader(data):
    nibabel_reader = NibabelReader()
    transform = Compose([
        LoadImaged(keys=["image", "label"], reader=nibabel_reader, image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ])
    dataset = Dataset(data=data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)
    return dataloader

def create_test_loader(data):
    nibabel_reader = NibabelReader()
    transform = Compose([
        LoadImaged(keys=["image"], reader=nibabel_reader, image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        ToTensord(keys=["image"]),
    ])
    dataset = Dataset(data=data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)
    return dataloader
